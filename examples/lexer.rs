extern crate symb;

use symb::*;

fn main() {
    let s = "5*x + 7*y^2 / (z+3)^3";
    let tokens = lex(s).unwrap();
    println!("{:?}", tokens);
}