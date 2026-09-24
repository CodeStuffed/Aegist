//! council-engine: a claim goes in and is judged by a council of personas
//! running on a transformer written and trained from scratch.

pub mod brain;
pub mod checkpoint;
pub mod config;
pub mod corpus;
pub mod council;
pub mod hardware;
pub mod kernels;
pub mod knowledge;
pub mod model;
pub mod optim;
pub mod quant;
pub mod research;
pub mod rng;
pub mod router;
pub mod session_log;
pub mod text;
pub mod tokenizer;
pub mod trainer;
pub mod util;
pub mod wikipedia;
