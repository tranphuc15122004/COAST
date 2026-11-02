# vrpr

Code and test data for the paper "Evolving routing and sequencing policies for dynamic vehicle routing problem with time windows".

Specify configuration in `.env` file, like so:
```env
LOG_HEU=stdout
# LOG_SIM=stdout
LOG_GP=stdout
LOG_LASTPOP=stdout
# LOG_ROUTE=stdout
LOG_LASTROUTE=stdout
LOG_DEBUG=stdout
POP_SIZE=100
NUM_GEN=100
MAX_DEPTH=6
WEIGHT=0.5
CROSSOVER_RATE=0.8
MUTATION_RATE=0.15
NUM_TIME_SLOT=20
TRAIN_FACTOR=2
STRESS_FACTOR=1
CONST_RATE=0.0
```

To run, execute:
```sh
# debug mode
cargo run -- [path to csv test file]

# release-lto mode (best performance)
cargo run --profile release-lto -- [path to csv test file]
```

Output log is formatted in structured JSONL format. Use a tool like [jq](https://jqlang.github.io/jq/) to extract relevant data.
