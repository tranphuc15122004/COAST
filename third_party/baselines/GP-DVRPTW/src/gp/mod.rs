use std::{cell::RefCell, ops::Range};

use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng, RngCore,
};

use crate::CONST_RATE;

use self::program::{Node, Program, ProgramContext, MAX_PROGRAM_NODE_CHILDREN};

pub mod program;

pub struct GPContext<R: RngCore> {
    pub rng: RefCell<R>,
    pub num_population: usize,
    pub max_depth: usize,
}

impl<R: RngCore> GPContext<R> {
    pub fn gen_terminal_at<C: ProgramContext>(&self, program: &mut Program<C>, index: usize) {
        let terminal = self.rng.borrow_mut().gen_bool(1.0 - *CONST_RATE);
        if terminal {
            let term_index = self.rng.borrow_mut().gen_range(0..C::num_terminals());
            program.generate_at(index, 0, Node::Terminal(term_index).into(), |_, _, _| {})
        } else {
            program.generate_at(
                index,
                0,
                self.rng.borrow_mut().gen_range(0u8 ..= 8) * 16,
                |_, _, _| {},
            )
        }
    }

    pub fn gen_internal_at<C: ProgramContext>(
        &self,
        program: &mut Program<C>,
        index: usize,
        gen_child_fn: impl FnMut(&mut Program<C>, usize, usize),
    ) {
        let int_index = self.rng.borrow_mut().gen_range(0..C::num_internals());
        program.generate_at(
            index,
            C::internal_num_children(int_index),
            Node::Internal(int_index).into(),
            gen_child_fn,
        );
    }

    pub fn gen_generic_at<C: ProgramContext>(
        &self,
        program: &mut Program<C>,
        index: usize,
        gen_child_fn: impl FnMut(&mut Program<C>, usize, usize),
    ) {
        let type_index = self
            .rng
            .borrow_mut()
            .gen_range(0..(C::num_internals() + C::num_terminals()));
        let (value, num_children) = if type_index < C::num_terminals() {
            (Node::Terminal(type_index).into(), 0)
        } else {
            (
                Node::Internal(type_index - C::num_terminals()).into(),
                C::internal_num_children(type_index - C::num_terminals()),
            )
        };
        program.generate_at(index, num_children, value, gen_child_fn);
    }

    pub fn gen_full_at<C: ProgramContext>(
        &self,
        program: &mut Program<C>,
        index: usize,
        depth: usize,
    ) {
        if depth == 0 {
            self.gen_terminal_at(program, index);
        } else {
            self.gen_internal_at(program, index, |program, _, index| {
                self.gen_full_at(program, index, depth - 1);
            });
        }
    }

    pub fn gen_grow_at<C: ProgramContext>(
        &self,
        program: &mut Program<C>,
        index: usize,
        depth: usize,
    ) {
        if depth == 0 {
            self.gen_terminal_at(program, index);
        } else {
            self.gen_generic_at(program, index, |program, _, index| {
                self.gen_grow_at(program, index, depth - 1);
            });
        }
    }

    // 0 -> 0; 1,2 -> 1; 3,4,5,6 -> 2; etc.
    fn depth_from_top(index: usize) -> usize {
        (index + 1).ilog2().try_into().unwrap()
    }

    fn depth_to_bottom<C: ProgramContext>(program: &Program<C>, index: usize) -> usize {
        match Node::from(program.nodes[index]) {
            Node::Internal(i) => Program::<C>::child_indices(index, C::internal_num_children(i))
                .map(|child_index| Self::depth_to_bottom(program, child_index) + 1)
                .max()
                .unwrap_or_default(),
            _ => 0,
        }
    }

    pub fn mutation<C: ProgramContext>(&self, p: &Program<C>) -> Program<C> {
        let mut p: Program<C> = p.clone();
        let swap_pos = *p
            .all_active_indices()
            .choose(&mut *self.rng.borrow_mut())
            .unwrap();
        p.clear_subtree(swap_pos);
        self.gen_grow_at(
            &mut p,
            swap_pos,
            self.max_depth - Self::depth_from_top(swap_pos),
        );
        p.verify();
        p
    }

    fn copy_subtree<C: ProgramContext>(
        dest: &mut Program<C>,
        dest_index: usize,
        src: &Program<C>,
        src_index: usize,
    ) {
        let num_children = match Node::from(src.nodes[src_index]) {
            Node::Internal(i) => C::internal_num_children(i),
            _ => 0,
        };
        dest.generate_at(
            dest_index,
            num_children,
            src.nodes[src_index],
            |dest, i, dest_child_index| {
                Self::copy_subtree(
                    dest,
                    dest_child_index,
                    src,
                    src_index * MAX_PROGRAM_NODE_CHILDREN + i + 1,
                );
            },
        )
    }

    fn all_index_of_layer(layer: usize) -> Range<usize> {
        ((1 << layer) - 1)..(1 << (layer + 1)) - 1
    }

    pub fn crossover<'a, C: ProgramContext>(
        &self,
        p1: &'a Program<C>,
        p2: &'a Program<C>,
    ) -> (Program<C>, Program<C>) {
        let mut c1 = p1.clone();
        let mut c2 = p2.clone();
        let depth1 = Self::depth_to_bottom(&c1, 0);
        let depth2 = Self::depth_to_bottom(&c2, 0);

        let depth_point1 = self.rng.borrow_mut().gen_range(0..=depth1);
        let min_depth_point2 = (depth_point1 + depth2).saturating_sub(self.max_depth);
        let max_depth_point2 = (self.max_depth - depth1 + depth_point1).min(depth2);

        let depth_point2 = self
            .rng
            .borrow_mut()
            .gen_range(min_depth_point2..=max_depth_point2);

        let swap_idx1 = Self::all_index_of_layer(depth_point1)
            .filter(|i| *i < c1.nodes.len() && !Node::from(c1.nodes[*i]).is_null())
            .choose(&mut *self.rng.borrow_mut())
            .expect("should not be None");
        let swap_idx2 = Self::all_index_of_layer(depth_point2)
            .filter(|i| *i < c2.nodes.len() && !Node::from(c2.nodes[*i]).is_null())
            .choose(&mut *self.rng.borrow_mut())
            .expect("should not be None");

        c1.clear_subtree(swap_idx1);
        c2.clear_subtree(swap_idx2);

        Self::copy_subtree(&mut c1, swap_idx1, p2, swap_idx2);
        Self::copy_subtree(&mut c2, swap_idx2, p1, swap_idx1);
        assert!(Self::depth_to_bottom(&c1, 0) <= self.max_depth);
        assert!(Self::depth_to_bottom(&c2, 0) <= self.max_depth);

        c1.verify();
        c2.verify();

        (c1, c2)
    }

    pub fn ramp_half_and_half<C: ProgramContext>(&self) -> Vec<Program<C>> {
        let mut v = Vec::new();
        let half_size = self.num_population / 2;
        for depth in 1..self.max_depth {
            for _ in 0..half_size / self.max_depth {
                let mut p = Program::new();
                self.gen_full_at(&mut p, 0, depth);
                p.verify();
                v.push(p);
                let mut p = Program::new();
                self.gen_grow_at(&mut p, 0, depth);
                p.verify();
                v.push(p);
            }
        }

        while v.len() < self.num_population {
            let mut p = Program::new();
            self.gen_grow_at(&mut p, 0, self.max_depth);
            p.verify();
            v.push(p);
        }

        v
    }
}

#[test]
fn tst() {
    use rand::rngs::ThreadRng;
    assert_eq!(GPContext::<ThreadRng>::all_index_of_layer(0), 0..1);
    assert_eq!(GPContext::<ThreadRng>::all_index_of_layer(1), 1..3);
    assert_eq!(GPContext::<ThreadRng>::all_index_of_layer(2), 3..7);
}
