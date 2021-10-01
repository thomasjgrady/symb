extern crate symb;

use std::str::FromStr;

use symb::*;

fn main() {
    let mut expression = Expression::from_str("5.73*x + 7.123*y^2 / (z+3.2)^3").unwrap();
    let target = Expression::from_str("y").unwrap();
    let replacement = Expression::from_str("(x^2 * (3y+z))").unwrap();

    println!("BEFORE: {}", expression);
    expression.subs(&target, &replacement);
    println!("AFTER: {}", expression);
}