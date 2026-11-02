use std::{env::var, fmt::Display, fs::File, io::Write, sync::Mutex};

use chrono::Local;
use miniserde::{json, Serialize};

#[macro_export]
macro_rules! log {
    ($target: expr, $arg:expr) => {
        $target.begin();
        $target.log_message($arg);
        $target.end();
    };
    ($target: expr, $arg:expr, $($key:ident = $value:expr),*) => {
        $target.begin();
        $target.log_message($arg);
        $(
            $target.log_key_value(stringify!($key), &$value, true);
        )*
        $target.end();
    };
}

pub enum LogTarget {
    Stdout,
    Stderr,
    File(Mutex<File>),
}

fn now() -> i64 {
    Local::now()
        .timestamp_nanos_opt()
        .expect("out of range datetime")
}

impl LogTarget {
    pub fn parse(str: &str) -> Option<LogTarget> {
        if str.trim().is_empty() {
            return None;
        }

        Some(match str {
            "stdout" => LogTarget::Stdout,
            "stderr" => LogTarget::Stderr,
            str => LogTarget::File(Mutex::new(File::open(str).expect("fail to open log file"))),
        })
    }
}

pub struct Logger {
    name: String,
    target: Option<LogTarget>,
}

impl Logger {
    fn args(&self, value: impl Display) -> String {
        let time = Local::now().format("%H:%M:%S%.9f");
        let name = &self.name;
        format!("{time},{name},{value}")
    }

    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            target: LogTarget::parse(&var(format!("LOG_{name}")).unwrap_or_default()),
        }
    }

    fn write(&self, value: impl Display) {
        match &self.target {
            Some(LogTarget::Stdout) => print!("{}", value),
            Some(LogTarget::Stderr) => eprint!("{}", value),
            Some(LogTarget::File(file)) => {
                let mut file = file.lock().expect("mutex lock failure");
                write!(&mut file, "{}", value).expect("write failed");
            }
            None => {}
        }
    }

    fn new_line(&self) {
        self.write('\n');
    }

    pub fn begin(&self) {
        self.write('{');
        self.log_key_value("__", &self.name, false);
        self.log_key_value("_t", &now(), true);
    }

    pub fn end(&self) {
        self.write('}');
        self.new_line();
    }

    pub fn log_message(&self, msg: impl Display) {
        self.log_key_value("_", &msg.to_string(), true);
    }

    pub fn log_key_value(&self, key: &str, value: &impl Serialize, comma: bool) {
        if comma {
            self.write(',');
        }
        self.write(json::to_string(key));
        self.write(':');
        self.write(json::to_string(value));
    }

    pub fn log(&self, value: impl Display) {
        let value = self.args(value);
        match &self.target {
            Some(LogTarget::Stdout) => println!("{}", value),
            Some(LogTarget::Stderr) => eprintln!("{}", value),
            Some(LogTarget::File(file)) => {
                let mut file = file.lock().expect("mutex lock failure");
                writeln!(&mut file, "{}", value).expect("write failed");
            }
            None => {}
        }
    }

    pub fn enabled(&self) -> bool {
        self.target.is_some()
    }
}
