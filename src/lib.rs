//! Aegist: a coding AI written and trained from scratch - its own
//! transformer, tokenizer and training loop, no outside models - plus the
//! tools around it that check its work and say so when it can't.

pub mod actions;
pub mod app;
pub mod assist;
pub mod bench;
pub mod brain;
pub mod checkpoint;
pub mod config;
pub mod corpus;
pub mod gpu;
pub mod ide;
pub mod hardware;
pub mod kernels;
pub mod lang;
pub mod learn;
pub mod model;
pub mod optim;
pub mod proc;
pub mod project;
pub mod prompt;
pub mod quant;
pub mod rng;
pub mod session;
pub mod tokenizer;
pub mod ui;
pub mod trainer;
pub mod util;
pub mod verify;
