extern crate symb;

use symb::*;

fn main() {
    let s = "5.73*x + 7.123*y^2 / (z+3.2)^3";
    let tokens = lex(s).unwrap();
    let expression = parse(&tokens).unwrap();

    println!("{}", expression);
}