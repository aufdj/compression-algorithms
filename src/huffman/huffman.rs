use std::cmp::Ordering;

#[derive(Eq, PartialEq)]
pub enum NodeType {
    Internal(Box<Node>, Box<Node>),
    Leaf(u8),
}

#[derive(Eq, PartialEq)]
pub struct Node {
    pub frequency: u32,
    pub node_type: NodeType,
}

impl Node {
    pub fn new(frequency: u32, node_type: NodeType) -> Node {
        Node { 
            frequency, 
            node_type 
        }
    }
}

impl Ord for Node {
    fn cmp(&self, rhs: &Self) -> Ordering {
        rhs.frequency.cmp(&self.frequency)
    }
} 

impl PartialOrd for Node {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
