use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::iter::Peekable;
use std::ops::{Add, Sub, Mul, Div};
use std::str::Chars;
use std::str::FromStr;

use itertools::Itertools;
use itertools::structs::Unique;

#[derive(Clone, Debug, PartialEq)]
pub enum Side {
    Left,
    Right
}

#[derive(Clone, Debug, PartialEq)]
pub enum Separator {
    Paren(Side),
    Brace(Side)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Delimiter {
    Comma,
    Semicolon
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Exp
}

impl Operator {
    fn precedence(&self) -> usize {
        match self {
            Self::Add => 1,
            Self::Sub => 1,
            Self::Mul => 2,
            Self::Div => 2,
            Self::Exp => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Ident(String),
    Literal(f64),
    Separator(Separator),
    Delimiter(Delimiter),
    Operator(Operator)
}

pub fn lex(s: &str) -> Option<Vec<Token>> {
    let mut iter = s.chars().peekable();
    let mut tokens = Vec::new();

    while let Some(c) = iter.peek() {
        if c.is_whitespace() { iter.next(); }
        else if c.is_alphabetic() { tokens.push(lex_ident(&mut iter)?); }
        else if c.is_digit(10) || *c == '.' { tokens.push(lex_literal(&mut iter)?); }
        else {
            match c {
                '(' => tokens.push(Token::Separator(Separator::Paren(Side::Left))),
                ')' => tokens.push(Token::Separator(Separator::Paren(Side::Right))),
                '[' => tokens.push(Token::Separator(Separator::Brace(Side::Left))),
                ']' => tokens.push(Token::Separator(Separator::Brace(Side::Right))),

                ',' => tokens.push(Token::Delimiter(Delimiter::Comma)),
                ';' => tokens.push(Token::Delimiter(Delimiter::Semicolon)),

                '+' => tokens.push(Token::Operator(Operator::Add)),
                '-' => tokens.push(Token::Operator(Operator::Sub)),
                '*' => tokens.push(Token::Operator(Operator::Mul)),
                '/' => tokens.push(Token::Operator(Operator::Div)),
                '^' => tokens.push(Token::Operator(Operator::Exp)),

                _ => return None
            }
            iter.next();
        }
    }

    Some(tokens)
}

fn lex_ident(iter: &mut Peekable<Chars<'_>>) -> Option<Token> {
    let mut collector = Vec::new();
    while let Some(c) = iter.next() {
        collector.push(c);
        if let Some(cn) = iter.peek() {
            if !cn.is_alphanumeric() { break; }
        }
    }
    Some(Token::Ident(collector.into_iter().collect()))
}

fn lex_literal(iter: &mut Peekable<Chars<'_>>) -> Option<Token> {
    let mut collector = Vec::new();
    while let Some(c) = iter.next() {
        collector.push(c);
        if let Some(cn) = iter.peek() {
            if !(cn.is_digit(10) || *cn == '.') {
                break;
            }
        }
    }
    let literal_string: String = collector.into_iter().collect();
    let literal = f64::from_str(&literal_string).ok()?;
    Some(Token::Literal(literal))
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expression {
    Ident(String),
    Literal(f64),
    Paren(Box<Expression>),
    Vector(Vec<Expression>),
    Matrix(Vec<Expression>, Vec<usize>),
    Operator(Operator, Vec<Expression>)
}

pub fn parse(tokens: &Vec<Token>) -> Option<Expression> {
    descend(tokens.as_slice())
}

fn descend(tokens: &[Token]) -> Option<Expression> {

    let mut op_data: Option<(usize, Operator)> = None;
    let mut paren_depth = 0;
    let mut brace_depth = 0;

    for (index, token) in tokens.iter().enumerate() {
        match token {
            Token::Separator(s) => {
                match s {
                    Separator::Paren(Side::Left) => paren_depth += 1,
                    Separator::Paren(Side::Right) => paren_depth -= 1,
                    Separator::Brace(Side::Left) => brace_depth += 1,
                    Separator::Brace(Side::Right) => brace_depth -= 1,
                }
            },
            Token::Operator(op) => {
                if paren_depth == 0 && brace_depth == 0 {
                    if let Some((_i, o)) = op_data {
                        if op.precedence() < o.precedence() {
                            op_data = Some((index, *op));
                        }
                    } else {
                        op_data = Some((index, *op));
                    }
                }
            },
            _ => ()
        }
    }

    if let Some((index, op)) = op_data {
        let left = &tokens[..index];
        let right = &tokens[(index+1)..];
        if left.len() == 0 || right.len() == 0 { return None; }

        let e1 = descend(left)?;
        let e2 = descend(right)?;
        return Some(Expression::Operator(op, vec![e1, e2]));
    }

    let token = tokens.get(0)?;
    match token {
        Token::Literal(f) => Some(Expression::Literal(*f)),
        Token::Ident(s) => Some(Expression::Ident(s.clone())),
        Token::Separator(s) => {
            match s {
                Separator::Paren(Side::Left) => {
                    let inner = descend(&tokens[1..(tokens.len()-1)])?;
                    Some(Expression::Paren(Box::new(inner)))            
                },
                _ => None
            }
        }
        _ => None
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expression::Ident(s) => write!(f, "{}", s),
            Expression::Literal(x) => write!(f, "{}", x),
            Expression::Paren(e) => write!(f, "({})", e),
            Expression::Vector(v) => {
                write!(f, "[")?;
                for i in 0..v.len() {
                    if i < v.len()-1 { write!(f, "{}, ", v[i])?; }
                    else { write!(f, "{}", v[i])?; }
                }
                write!(f, "]")
            },
            Expression::Matrix(v, shape) => {
                write!(f, "[")?;
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        if j < shape[1]-1 { write!(f, "{} ,", v[i*shape[1] + j])?; }
                        else { write!(f, "{}", v[i*shape[1] + j])?; }
                    }
                    if i < shape[0]-1 { write!(f, ";")?; }
                }
                write!(f, "]")
            },
            Expression::Operator(op, args) => {
                write!(f, "{}", args[0])?;
                match op {
                    Operator::Add => write!(f, " + ")?,
                    Operator::Sub => write!(f, " - ")?,
                    Operator::Mul => write!(f, " * ")?,
                    Operator::Div => write!(f, " / ")?,
                    Operator::Exp => write!(f, " ^ ")?,
                }
                write!(f, "{}", args[1])
            }
        }
    }
}

impl FromStr for Expression {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(tokens) = lex(s) {
            if let Some(e) = parse(&tokens) {
                return Ok(e)
            }
        }
        Err(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExpressionReturnType {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix(Vec<f64>, Vec<usize>)
}

impl Add for ExpressionReturnType {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Scalar(x), Self::Scalar(y)) => Self::Scalar(x+y),
            (Self::Scalar(x), Self::Vector(v)) => Self::Vector(v.iter().map(|y| x+y).collect()),
            (Self::Scalar(x), Self::Matrix(v, shape)) => Self::Matrix(v.iter().map(|y| x+y).collect(), shape),
            (Self::Vector(x), Self::Vector(y)) => Self::Vector(x.iter().zip(y.iter()).map(|(a, b)| a+b).collect()),
            (Self::Matrix(x, s1), Self::Matrix(y, s2)) => {
                assert!(s1 == s2);
                Self::Matrix(x.iter().zip(y.iter()).map(|(a, b)| a+b).collect(), s1)
            },
            (Self::Vector(x), Self::Matrix(y, shape)) => panic!("Cannot add vector to matrix elementwise"),
            (r1, r2) => r2.add(r1)
        }
    }
}

impl Sub for ExpressionReturnType {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Scalar(x), Self::Scalar(y)) => Self::Scalar(x-y),
            (Self::Scalar(x), Self::Vector(v)) => Self::Vector(v.iter().map(|y| x-y).collect()),
            (Self::Scalar(x), Self::Matrix(v, shape)) => Self::Matrix(v.iter().map(|y| x-y).collect(), shape),
            (Self::Vector(x), Self::Vector(y)) => Self::Vector(x.iter().zip(y.iter()).map(|(a, b)| a-b).collect()),
            (Self::Matrix(x, s1), Self::Matrix(y, s2)) => {
                assert!(s1 == s2);
                Self::Matrix(x.iter().zip(y.iter()).map(|(a, b)| a-b).collect(), s1)
            },
            (Self::Vector(x), Self::Matrix(y, shape)) => panic!("Cannot sub vector to matrix elementwise"),
            (r1, r2) => r2.sub(r1)
        }
    }
}

impl Mul for ExpressionReturnType {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Scalar(x), Self::Scalar(y)) => Self::Scalar(x*y),
            (Self::Scalar(x), Self::Vector(v)) => Self::Vector(v.iter().map(|y| x*y).collect()),
            (Self::Scalar(x), Self::Matrix(v, shape)) => Self::Matrix(v.iter().map(|y| x*y).collect(), shape),
            (Self::Vector(x), Self::Vector(y)) => Self::Vector(x.iter().zip(y.iter()).map(|(a, b)| a*b).collect()),
            (Self::Matrix(x, s1), Self::Matrix(y, s2)) => {
                assert!(s1 == s2);
                Self::Matrix(x.iter().zip(y.iter()).map(|(a, b)| a*b).collect(), s1)
            },
            (Self::Vector(x), Self::Matrix(y, shape)) => panic!("Cannot mul vector to matrix elementwise"),
            (r1, r2) => r2.mul(r1)
        }
    }
}

impl Div for ExpressionReturnType {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Scalar(x), Self::Scalar(y)) => Self::Scalar(x/y),
            (Self::Scalar(x), Self::Vector(v)) => Self::Vector(v.iter().map(|y| x/y).collect()),
            (Self::Scalar(x), Self::Matrix(v, shape)) => Self::Matrix(v.iter().map(|y| x/y).collect(), shape),
            (Self::Vector(x), Self::Vector(y)) => Self::Vector(x.iter().zip(y.iter()).map(|(a, b)| a/b).collect()),
            (Self::Matrix(x, s1), Self::Matrix(y, s2)) => {
                assert!(s1 == s2);
                Self::Matrix(x.iter().zip(y.iter()).map(|(a, b)| a/b).collect(), s1)
            },
            (Self::Vector(x), Self::Matrix(y, shape)) => panic!("Cannot div vector to matrix elementwise"),
            (r1, r2) => r2.div(r1)
        }
    }
}

impl ExpressionReturnType {
    fn exp(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Scalar(x), Self::Scalar(y)) => Self::Scalar(x.powf(y)),
            (Self::Scalar(x), Self::Vector(v)) => Self::Vector(v.iter().map(|y| x.powf(*y)).collect()),
            (Self::Scalar(x), Self::Matrix(v, shape)) => Self::Matrix(v.iter().map(|y| x.powf(*y)).collect(), shape),
            (Self::Vector(x), Self::Vector(y)) => Self::Vector(x.iter().zip(y.iter()).map(|(a, b)| a.powf(*b)).collect()),
            (Self::Matrix(x, s1), Self::Matrix(y, s2)) => {
                assert!(s1 == s2);
                Self::Matrix(x.iter().zip(y.iter()).map(|(a, b)| a.powf(*b)).collect(), s1)
            },
            (Self::Vector(x), Self::Matrix(y, shape)) => panic!("Cannot exp vector to matrix elementwise"),
            (r1, r2) => r2.exp(r1)
        }
    }
}

pub type ExpressionKernel = Box<dyn Fn(&Vec<f64>) -> ExpressionReturnType>;

impl Expression {
    pub fn subs(&mut self, target: &Expression, replacement: &Expression) -> Option<()> {
        if self == target {
            *self = Self::Paren(Box::new(replacement.clone()));
            return Some(());
        }

        match self {
            Self::Paren(e) => e.subs(target, replacement),
            Self::Vector(v) => {
                for e in v.iter_mut() {
                    if let Some(()) = e.subs(target, replacement) {
                        return Some(());
                    }
                }
                None
            },
            Self::Matrix(v, _shape) => {
                for e in v.iter_mut() {
                    if let Some(()) = e.subs(target, replacement) {
                        return Some(());
                    }
                }
                None
            },
            Self::Operator(_op, args) => {
                for e in args.iter_mut() {
                    if let Some(()) = e.subs(target, replacement) {
                        return Some(());
                    }
                }
                None
            },
            _ => None
        }
    }

    pub fn kernel(&self) -> ExpressionKernel {
        let mut ident_vec = Vec::new();
        self.find_idents(&mut ident_vec);
        
        let ident_map: HashMap<String, usize> = ident_vec.iter()
            .unique()
            .sorted()
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        self._kernel(&ident_map)
    }

    fn find_idents(&self, ident_vec: &mut Vec<String>) {
        match self {
            Self::Ident(s) => ident_vec.push(s.clone()),
            Self::Literal(_) => (),
            Self::Paren(e) => e.find_idents(ident_vec),
            Self::Vector(v) => {
                for e in v {
                    e.find_idents(ident_vec)
                }
            },
            Self::Matrix(v, _s) => {
                for e in v {
                    e.find_idents(ident_vec)
                }
            },
            Self::Operator(op, args) => {
                for e in args {
                    e.find_idents(ident_vec)
                }
            }
        }
    }

    fn _kernel(&self, ident_map: &HashMap<String, usize>) -> ExpressionKernel {
        match self {
            Self::Ident(s) => {
                let index = ident_map.get(s).unwrap().clone();
                Box::new(move |args| ExpressionReturnType::Scalar(args[index]))
            },
            Self::Literal( x) => {
                let y = x.clone();
                Box::new(move |_args| ExpressionReturnType::Scalar(y))
            },
            Self::Paren(e) => e._kernel(ident_map),
            Self::Vector(v) => {
                let kernels: Vec<_> = v.iter()
                    .map(|e| e._kernel(ident_map))
                    .collect();

                Box::new(move |args| {
                    let results = kernels.iter()
                        .map(|k| k(args))
                        .map(|ret| {
                            match ret {
                                ExpressionReturnType::Scalar(x) => x,
                                _ => panic!("Subexpressions of vectors must return scalars")
                            }
                        })
                        .collect();
                    ExpressionReturnType::Vector(results)
                })
            },
            Self::Matrix(v, shape) => unimplemented!(),
            Self::Operator(op, v) => {
                let kernels: Vec<_> = v.iter()
                    .map(|e| e._kernel(ident_map))
                    .collect();
                
                match op {
                    Operator::Add => Box::new(move |args| kernels[0](args) + kernels[1](args)),
                    Operator::Sub => Box::new(move |args| kernels[0](args) - kernels[1](args)),
                    Operator::Mul => Box::new(move |args| kernels[0](args) * kernels[1](args)),
                    Operator::Div => Box::new(move |args| kernels[0](args) / kernels[1](args)),
                    Operator::Exp => Box::new(move |args| kernels[0](args).exp(kernels[1](args))),
                }
            }
        }
    }
}