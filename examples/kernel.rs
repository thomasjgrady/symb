extern crate symb;

use std::str::FromStr;

use symb::*;

fn main() {
    let mut expression = Expression::from_str("(x+y)/z").unwrap();
    let kernel = expression.kernel();

    let args: Vec<f64> = vec![2.0, 4.0, 3.0];
    let result = kernel(&args);
    println!("{:?}", result);
}