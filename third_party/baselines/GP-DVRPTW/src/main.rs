use std::{
    cell::RefCell,
    env::{self, args},
};

use gp::{program::Node, GPContext};
use lazy_static::lazy_static;
use log::Logger;
use lru::LruCache;
use ordered_float::OrderedFloat;
use rand::{rngs::SmallRng, Rng, RngCore, SeedableRng};
use sim::{
    ctx::{RoutingProgram, SequencingProgram},
    problem::Problem,
    Simulation,
};

pub mod gp;
pub mod log;
pub mod sim;

lazy_static! {
    static ref MAIN: Logger = Logger::new("MAIN");
    static ref HEU: Logger = Logger::new("HEU");
    static ref SIM: Logger = Logger::new("SIM");
    static ref GP: Logger = Logger::new("GP");
    static ref LASTPOP: Logger = Logger::new("LASTPOP");
    static ref LASTROUTE: Logger = Logger::new("LASTROUTE");
    static ref ROUTE: Logger = Logger::new("ROUTE");
    static ref ROUTEEVAL: Logger = Logger::new("ROUTEEVAL");
    static ref DEBUG: Logger = Logger::new("DEBUG");
    static ref CONST_RATE: f64 = env::var("CONST_RATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    static ref WEIGHT: f32 = env::var("WEIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    static ref NUM_TIME_SLOT: f32 = env::var("NUM_TIME_SLOT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);
    static ref NUM_GEN: usize = env::var("NUM_GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    static ref POP_SIZE: usize = env::var("POP_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    static ref MAX_DEPTH: usize = env::var("MAX_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    static ref CROSSOVER_RATE: f64 = env::var("CROSSOVER_RATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.8);
    static ref MUTATION_RATE: f64 = env::var("MUTATION_RATE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    static ref TRAIN_FACTOR: f32 = env::var("TRAIN_FACTOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.2);
    static ref STRESS_FACTOR: f32 = env::var("STRESS_FACTOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
}

fn fitness(problem: &Problem, result: (f32, usize)) -> f32 {
    let (distance, num_fail) = result;
    let tot_dist = problem.truck_speed * problem.depot.close * problem.num_trucks as f32;
    let weight = *WEIGHT;
    distance / tot_dist * weight
        + (num_fail as f32) / (problem.requests.len() as f32) * (1.0 - weight)
}

#[allow(non_snake_case)]
fn heuristics(problem: &Problem) -> anyhow::Result<()> {
    let CR = RoutingProgram::terminal(3);
    let CS = SequencingProgram::from_vec(vec![
        Node::Internal(5).into(),
        Node::Terminal(0).into(),
        Node::Terminal(4).into(),
    ]);
    let W = SequencingProgram::terminal(3);
    let WIQ = RoutingProgram::terminal(1);
    for (name, r, s) in [("C+C", &CR, &CS), ("C+W", &CR, &W), ("WIQ+C", &WIQ, &CS)] {
        let mut simulation = Simulation::new(problem, r, s);
        let result = simulation.simulate_until(problem.depot.close / *NUM_TIME_SLOT, f32::MAX);
        log!(
            HEU,
            "heuristic_result",
            name = name,
            result = result,
            fitness = fitness(problem, result)
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Individual<'a> {
    routing: RoutingProgram<'a>,
    sequencing: SequencingProgram<'a>,
    pub result: Option<(f32, usize, f32)>,
}

impl<'a> Individual<'a> {
    pub fn ramp_half_and_half(gpc: &GPContext<impl RngCore>) -> Vec<Self> {
        let r_pop = gpc.ramp_half_and_half();
        let s_pop = gpc.ramp_half_and_half();
        r_pop
            .into_iter()
            .zip(s_pop)
            .map(|(routing, sequencing)| Self {
                routing,
                sequencing,
                result: None,
            })
            .collect()
    }

    pub fn crossover_with(&self, gpc: &GPContext<impl RngCore>, other: &Self) -> (Self, Self) {
        let (r1, r2) = gpc.crossover(&self.routing, &other.routing);
        let (s1, s2) = gpc.crossover(&self.sequencing, &other.sequencing);
        (
            Self {
                routing: r1,
                sequencing: s1,
                result: None,
            },
            Self {
                routing: r2,
                sequencing: s2,
                result: None,
            },
        )
    }

    pub fn mutate(&self, gpc: &GPContext<impl RngCore>) -> Self {
        Self {
            routing: gpc.mutation(&self.routing),
            sequencing: gpc.mutation(&self.sequencing),
            result: None,
        }
    }

    pub fn evaluate(
        &mut self,
        cache: &mut LruCache<String, (f32, usize, f32)>,
        problem: &Problem,
        time_slot: f32,
    ) -> f32 {
        if let Some((_, _, fitness)) = self.result {
            return fitness;
        }

        let cache_key = format!("{}:{}", self.routing, self.sequencing);
        let result = *cache.get_or_insert(cache_key, || {
            let (dist, nb_fail) = Simulation::new(problem, &self.routing, &self.sequencing)
                .simulate_until(time_slot, f32::MAX);
            let fitness = fitness(problem, (dist, nb_fail));
            (dist, nb_fail, fitness)
        });

        self.result = Some(result);
        result.2
    }
}

fn select_parent<'a>(gpc: &GPContext<impl RngCore>, pop: &'a [Individual<'a>]) -> usize {
    rand::seq::index::sample(&mut *gpc.rng.borrow_mut(), pop.len(), 8)
        .into_iter()
        .max_by_key(|i| OrderedFloat(pop[*i].result.unwrap().2))
        .unwrap()
}

fn gp(problem: &Problem) -> anyhow::Result<()> {
    let time_slot = problem.depot.close / *NUM_TIME_SLOT;
    let train_time_slot = time_slot / *STRESS_FACTOR;
    let training_problem = problem.clone_training(time_slot * (*TRAIN_FACTOR), *STRESS_FACTOR);
    let gpc = GPContext {
        rng: RefCell::new(SmallRng::from_entropy()),
        num_population: *POP_SIZE,
        max_depth: *MAX_DEPTH,
    };
    let mut cache = LruCache::unbounded();
    let mut pop = Individual::ramp_half_and_half(&gpc);
    for gen in 1..=*NUM_GEN {
        for i in pop.iter_mut() {
            i.evaluate(&mut cache, &training_problem, train_time_slot);
        }

        pop.sort_unstable_by_key(|i| OrderedFloat(i.result.unwrap().2));
        pop.truncate(gpc.num_population);
        let result = pop[0].result.unwrap();

        log!(
            GP,
            "new_gen",
            gen = gen,
            result = (result.0, result.1),
            fitness = result.2,
            routing = pop[0].routing.to_string(),
            sequencing = pop[0].sequencing.to_string()
        );
        let mut sim = Simulation::new(problem, &pop[0].routing, &pop[0].sequencing);
        let result = sim.simulate_until(time_slot, f32::MAX);
        log!(
            GP,
            "full_result",
            result = result,
            fitness = fitness(problem, result)
        );

        log!(
            GP,
            "base64",
            routing = pop[0].routing.base64(),
            sequencing = pop[0].sequencing.base64()
        );

        if gen == *NUM_GEN {
            for vehicle in 0..problem.num_trucks {
                log!(
                    LASTROUTE,
                    "route_log",
                    vehicle = vehicle,
                    route = sim.vehicles[vehicle].route,
                    dropped = sim.vehicles[vehicle].dropped
                );
            }
            for i in pop.iter() {
                log!(
                    LASTPOP,
                    "lastpop",
                    routing = i.routing.to_string(),
                    sequencing = i.sequencing.to_string()
                );
            }
        }

        for _ in 0..gpc.num_population / 2 {
            let p1 = select_parent(&gpc, &pop[0..gpc.num_population]);
            let p2 = select_parent(&gpc, &pop[0..gpc.num_population]);

            let x = gpc.rng.borrow_mut().gen_range(0.0..=1.0);
            match x {
                x if x <= *CROSSOVER_RATE => {
                    let (c1, c2) = pop[p1].crossover_with(&gpc, &pop[p2]);
                    pop.push(c1);
                    pop.push(c2);
                }
                x if x <= *CROSSOVER_RATE + *MUTATION_RATE => {
                    let m1 = pop[p1].mutate(&gpc);
                    let m2 = pop[p2].mutate(&gpc);
                    pop.push(m1);
                    pop.push(m2);
                }
                _ => {
                    pop.push(pop[p1].clone());
                    pop.push(pop[p2].clone());
                }
            }
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    _ = dotenv::dotenv()?;
    log!(MAIN, "start");
    let path = args().nth(1).expect("usage: cargo run -- [problem path]");
    let problem = Problem::load(&path, 1.0, 1300.0, 10)?;
    if HEU.enabled() {
        log!(MAIN, "heu_start");
        heuristics(&problem)?;
    }
    if GP.enabled() {
        log!(MAIN, "gp_start");
        gp(&problem)?;
    }
    Ok(())
}
