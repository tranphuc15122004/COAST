use std::{
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Clone, Copy)]
pub struct Request {
    pub idx: usize,
    pub x: f32,
    pub y: f32,
    pub demand: f32,
    pub open: f32,
    pub close: f32,
    pub service_time: f32,
    pub time: f32,
}

#[derive(Clone)]
pub struct Problem {
    pub depot: Request,
    pub requests: Vec<Request>,
    pub truck_speed: f32,
    pub truck_capacity: f32,
    pub num_trucks: usize,
}

impl Problem {
    pub fn load(
        csv: &str,
        truck_speed: f32,
        truck_capacity: f32,
        num_trucks: usize,
    ) -> anyhow::Result<Problem> {
        let file = BufReader::new(File::open(csv)?);
        let mut requests = Vec::new();
        let lines = file.lines().skip(1);
        for (idx, line) in lines.enumerate() {
            let args = line?
                .split(',')
                .map(|tok| tok.parse::<f32>())
                .collect::<Result<Vec<f32>, _>>()?;
            let req = Request {
                idx,
                x: args[0],
                y: args[1],
                demand: args[2],
                open: args[3],
                close: args[4],
                service_time: 10.0,
                time: args[7],
            };
            requests.push(req);
        }
        let depot = requests.remove(0);
        Ok(Self {
            depot,
            requests,
            truck_speed,
            truck_capacity,
            num_trucks,
        })
    }

    pub fn clone_training(&self, time_limit: f32, stress_factor: f32) -> Self {
        let mut requests = Vec::new();
        let mut current_index = 0;
        let mut turn = 0.0f32;
        for mut req in self.requests.iter().cloned() {
            req.x *= stress_factor;
            req.y *= stress_factor;
            req.service_time *= stress_factor;
            if req.time > time_limit {
                let time_req = self.requests[current_index];
                req.time = time_limit * turn + (time_req.time + time_req.open * 1.5) / 2.5;
                req.open = time_limit * turn + time_req.open;
                req.close = time_limit * turn + time_req.close;

                current_index += 1;
                if self.requests[current_index].time > time_limit {
                    current_index = 0;
                    turn += 1.0;
                }
            }
            requests.push(req);
        }
        Self {
            depot: self.depot,
            requests,
            truck_speed: self.truck_speed,
            num_trucks: self.num_trucks,
            truck_capacity: self.truck_capacity,
        }
    }

    pub fn total_demand(&self) -> f32 {
        self.requests.iter().map(|r| r.demand).sum()
    }
}
