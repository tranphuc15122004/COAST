use std::fmt::{self, Formatter};

use smallvec::SmallVec;

use crate::gp::program::{Program, ProgramContext, MAX_PROGRAM_NODE_CHILDREN};

use super::{
    problem::{Problem, Request},
    VehicleState,
};

fn safe_div(x: f32, y: f32) -> f32 {
    if y.abs() < 1e-4 {
        1.0
    } else {
        x / y
    }
}

pub struct RoutingContext<'a> {
    pub vehicle_state: &'a VehicleState<'a>,
    pub problem: &'a Problem,
    pub time: f32,
    pub request: &'a Request,
}

pub struct SequencingContext<'a> {
    pub vehicle_state: &'a VehicleState<'a>,
    pub problem: &'a Problem,
    pub time: f32,
    pub request: &'a Request,
    pub ready_time: f32,
}

fn common_num_internal() -> usize {
    6
}

fn common_internal_num_children(_: usize) -> usize {
    2
}

fn common_format_terminal(index: usize, f: &mut Formatter<'_>) -> fmt::Result {
    write!(
        f,
        "{}",
        match index {
            0 => "sum",
            1 => "sub",
            2 => "mul",
            3 => "div",
            4 => "min",
            5 => "max",
            _ => unreachable!(),
        }
    )
}

fn common_internal(idx: usize, child_values: SmallVec<[f32; MAX_PROGRAM_NODE_CHILDREN]>) -> f32 {
    let x = child_values[0];
    let y = child_values[1];
    match idx {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => {
            if y.abs() < 1e-4 {
                1.0
            } else {
                x / y
            }
        }
        4 => x.min(y),
        5 => x.max(y),
        _ => unreachable!(),
    }
}

pub type RoutingProgram<'a> = Program<RoutingContext<'a>>;
pub type SequencingProgram<'a> = Program<SequencingContext<'a>>;

impl<'a> ProgramContext for RoutingContext<'a> {
    fn internal(
        &self,
        idx: usize,
        child_values: SmallVec<[f32; MAX_PROGRAM_NODE_CHILDREN]>,
    ) -> f32 {
        common_internal(idx, child_values)
    }

    fn internal_num_children(index: usize) -> usize {
        common_internal_num_children(index)
    }

    fn num_internals() -> usize {
        common_num_internal()
    }

    fn format_internal(index: usize, f: &mut Formatter<'_>) -> fmt::Result {
        common_format_terminal(index, f)
    }

    fn terminal(&self, idx: usize) -> f32 {
        match idx {
            0 => self.vehicle_state.queue.len() as f32 / self.problem.requests.len() as f32,
            1 => {
                (self.problem.truck_capacity
                    - self
                        .vehicle_state
                        .queue
                        .iter()
                        .map(|r| r.0.demand)
                        .sum::<f32>())
                    / self.problem.total_demand()
            }
            2 => {
                let (x, y) = self.vehicle_state.median_queue_pos();
                let (rx, ry) = (self.request.x, self.request.y);
                ((x - rx) * (x - rx) + (y - ry) * (y - ry)).sqrt()
                    / self.problem.truck_speed
                    / self.problem.depot.close
            }
            3 => {
                self.vehicle_state
                    .raw_time_cost(self.problem, self.request, self.time)
                    / self.problem.depot.close
            }
            4 => self.request.demand / self.problem.total_demand(),
            _ => unreachable!(),
        }
    }

    fn num_terminals() -> usize {
        5
    }
}

impl<'a> ProgramContext for SequencingContext<'a> {
    fn internal(
        &self,
        idx: usize,
        child_values: SmallVec<[f32; MAX_PROGRAM_NODE_CHILDREN]>,
    ) -> f32 {
        common_internal(idx, child_values)
    }

    fn internal_num_children(index: usize) -> usize {
        common_internal_num_children(index)
    }

    fn num_internals() -> usize {
        common_num_internal()
    }

    fn format_internal(index: usize, f: &mut Formatter<'_>) -> fmt::Result {
        common_format_terminal(index, f)
    }

    fn terminal(&self, idx: usize) -> f32 {
        let raw_time_cost = self
            .vehicle_state
            .raw_time_cost(self.problem, self.request, self.time);
        let time_until_close = self.request.close - self.vehicle_state.busy_until;
        let wait_time = self.time - self.request.open;
        match idx {
            0 => raw_time_cost / self.problem.depot.close,
            1 => (self.time - self.ready_time) / self.problem.depot.close,
            2 => safe_div(time_until_close - raw_time_cost, time_until_close),
            3 => self.request.demand / self.problem.total_demand(),
            4 => wait_time / self.problem.depot.close,
            5 => self.request.time / self.problem.depot.close,
            _ => unreachable!(),
        }
    }

    fn num_terminals() -> usize {
        6
    }
}
