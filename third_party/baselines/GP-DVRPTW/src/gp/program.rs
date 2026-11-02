use base64::{prelude::BASE64_STANDARD, Engine};
use core::f32;
use smallvec::SmallVec;
use std::{
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

pub const MAX_PROGRAM_NODE_CHILDREN: usize = 2;

pub trait ProgramContext {
    fn num_terminals() -> usize;
    fn num_internals() -> usize;
    fn internal_num_children(index: usize) -> usize;

    fn terminal(&self, index: usize) -> f32;
    fn internal(
        &self,
        index: usize,
        child_values: SmallVec<[f32; MAX_PROGRAM_NODE_CHILDREN]>,
    ) -> f32;

    fn format_terminal(index: usize, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "TERM{index}")
    }

    fn format_internal(index: usize, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "INT{index}")
    }
}

#[derive(Debug)]
pub enum Node {
    Const(f32),
    Terminal(usize),
    Internal(usize),
    Null,
}
impl Node {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

struct DisplayNode<'p, C: ProgramContext> {
    program: &'p Program<C>,
    index: usize,
}

impl<'p, C: ProgramContext> Display for DisplayNode<'p, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match Node::from(self.program.nodes[self.index]) {
            Node::Const(value) => write!(f, "{}", value)?,
            Node::Terminal(index) => C::format_terminal(index, f)?,
            Node::Internal(index) => {
                C::format_internal(index, f)?;
                write!(f, "(")?;
                for (i, child_index) in
                    Program::<C>::child_indices(self.index, C::internal_num_children(index))
                        .enumerate()
                {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{}",
                        DisplayNode {
                            program: self.program,
                            index: child_index
                        }
                    )?;
                }
                write!(f, ")")?;
            }
            Node::Null => write!(f, "INVALID")?,
        }
        Ok(())
    }
}

impl From<u8> for Node {
    fn from(x: u8) -> Self {
        match x {
            0..=128 => Self::Const(f32::from(i16::from(x) - 64) / 16.0),
            129..=192 => Self::Terminal((x - 129).into()),
            193..255 => Self::Internal((x - 193).into()),
            255 => Self::Null,
        }
    }
}

impl From<Node> for u8 {
    fn from(value: Node) -> Self {
        match value {
            Node::Const(x) => (x * 16.0 + 64.0) as u8,
            Node::Terminal(index) => (129 + index).try_into().unwrap(),
            Node::Internal(index) => (193 + index).try_into().unwrap(),
            Node::Null => 255,
        }
    }
}

pub struct Program<C: ProgramContext> {
    // 0-128: const values (normalized into x/16 - 1)
    // 129-192: terminals
    // 193-254: internals
    // 255: null
    pub nodes: Vec<u8>,
    _marker: PhantomData<fn() -> C>,
}

impl<C: ProgramContext> Default for Program<C> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            _marker: Default::default(),
        }
    }
}

impl<C: ProgramContext> Clone for Program<C> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            _marker: PhantomData,
        }
    }
}

impl<C: ProgramContext> Debug for Program<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Program")
            .field(
                "nodes",
                &self
                    .nodes
                    .iter()
                    .map(|val| Node::from(*val))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl<C: ProgramContext> Program<C> {
    pub fn new() -> Self {
        Self::from_vec(Default::default())
    }

    pub fn from_vec(nodes: Vec<u8>) -> Self {
        Self {
            nodes,
            _marker: PhantomData,
        }
    }

    pub fn terminal(index: usize) -> Self {
        Self {
            nodes: vec![Node::Terminal(index).into()],
            _marker: PhantomData,
        }
    }

    pub fn child_indices(index: usize, num_children: usize) -> impl Iterator<Item = usize> {
        (0..num_children).map(move |i| index * MAX_PROGRAM_NODE_CHILDREN + i + 1)
    }

    pub fn ensure_index_exist(&mut self, index: usize) {
        self.nodes
            .resize(self.nodes.len().max(index + 1), Node::Null.into());
    }

    pub fn generate_at(
        &mut self,
        index: usize,
        num_children: usize,
        value: u8,
        mut gen_child_fn: impl FnMut(&mut Self, usize, usize),
    ) {
        self.ensure_index_exist(index);
        self.nodes[index] = value;
        for (i, child_index) in Self::child_indices(index, num_children).enumerate() {
            gen_child_fn(self, i, child_index);
        }
    }

    fn calc_at(&self, c: &C, i: usize, term_cache: &[f32]) -> f32 {
        match Node::from(self.nodes[i]) {
            Node::Const(x) => x,
            Node::Terminal(idx) => term_cache[idx],
            Node::Internal(idx) => {
                let children: SmallVec<[f32; MAX_PROGRAM_NODE_CHILDREN]> =
                    Self::child_indices(i, C::internal_num_children(idx))
                        .map(|i| self.calc_at(c, i, term_cache))
                        .collect();
                c.internal(idx, children)
            }
            Node::Null => unreachable!(),
        }
    }

    pub fn calc(&self, c: &C) -> f32 {
        let term_cache = (0..C::num_terminals())
            .map(|i| c.terminal(i))
            .collect::<Vec<_>>();
        self.calc_at(c, 0, &term_cache)
    }

    pub fn collect_all_active_indices(&self, dest: &mut Vec<usize>, index: usize) {
        dest.push(index);
        if let Node::Internal(i) = Node::from(self.nodes[index]) {
            for child_index in Self::child_indices(index, C::internal_num_children(i)) {
                self.collect_all_active_indices(dest, child_index);
            }
        }
    }

    pub fn all_active_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        self.collect_all_active_indices(&mut indices, 0);
        indices
    }

    pub fn run_length_encode(v: &[u8]) -> Vec<u8> {
        let mut res: Vec<u8> = Vec::new();
        for byte in v {
            if res.len() < 2 || res[res.len() - 2] != *byte || res[res.len() - 1] == u8::MAX {
                res.push(*byte);
                res.push(0);
            } else {
                *res.last_mut().unwrap() += 1;
            }
        }
        res
    }

    pub fn run_length_decode(v: &[u8]) -> Vec<u8> {
        assert!(v.len() % 2 == 0);
        let mut res = Vec::new();
        for i in 0..v.len() / 2 {
            for _ in 0..v[i * 2 + 1] + 1 {
                res.push(v[i * 2]);
            }
        }
        res
    }

    pub fn base64(&self) -> String {
        BASE64_STANDARD.encode(Self::run_length_encode(&self.nodes))
    }

    pub fn from_base64(str: &str) -> Self {
        Self {
            nodes: Self::run_length_decode(&BASE64_STANDARD.decode(str).expect("invalid base64")),
            _marker: PhantomData,
        }
    }

    pub fn verify(&self) {
        debug_assert!({
            let all_actives = self.all_active_indices();
            self.nodes.iter().enumerate().any(|(idx, val)| {
                let is_null = matches!(Node::from(*val), Node::Null);
                let is_active = all_actives.contains(&idx);
                is_active != is_null
            })
        });
    }

    pub fn clear_subtree(&mut self, index: usize) {
        let mut indices = Vec::new();
        self.collect_all_active_indices(&mut indices, index);
        indices
            .into_iter()
            .for_each(|i| self.nodes[i] = Node::Null.into());
    }
}

impl<C: ProgramContext> Display for Program<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            DisplayNode {
                program: self,
                index: 0,
            }
        )
    }
}

#[test]
fn rle() {
    use crate::sim::ctx::SequencingContext;
    assert_eq!(
        &Program::<SequencingContext>::run_length_encode(&[1, 2, 3, 3, 3]),
        &[1, 0, 2, 0, 3, 2]
    );
    assert_eq!(
        Program::<SequencingContext>::run_length_decode(
            &Program::<SequencingContext>::run_length_encode(&[1, 2, 3, 3, 3])
        ),
        &[1, 2, 3, 3, 3]
    );
}
