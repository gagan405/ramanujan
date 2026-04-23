#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    RankMismatch {
        expect: usize,
        actual: usize,
    },
    ShapeMismatch {
        expect: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
}

impl std::error::Error for TensorError {}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::RankMismatch { expect, actual } => {
                write!(f, "rank mismatch: expected {expect}, got {actual}")
            }
            TensorError::ShapeMismatch { expect, actual } => write!(
                f,
                "shape mismatch: expected {expect} elements, got {actual}"
            ),
            TensorError::IndexOutOfBounds { index, shape } => {
                write!(f, "index {index:?} out of bounds for shape {shape:?}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> Tensor<T>
where
    T: Copy,
{
    pub fn new(data: Vec<T>, shape: Vec<usize>, strides: Vec<usize>) -> Result<Self, TensorError> {
        let expect: usize = shape.iter().product();
        if data.len() != expect {
            return Err(TensorError::ShapeMismatch {
                expect,
                actual: data.len(),
            });
        }
        if shape.len() != strides.len() {
            return Err(TensorError::RankMismatch {
                expect: shape.len(),
                actual: strides.len(),
            });
        }
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    pub fn index(&self, idx: &[usize]) -> Result<&T, TensorError> {
        if idx.len() != self.shape.len() {
            return Err(TensorError::RankMismatch {
                expect: self.shape.len(),
                actual: idx.len(),
            });
        }

        for (_dim, (&i, &dim_size)) in idx.iter().zip(self.shape.iter()).enumerate() {
            if i >= dim_size {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx.to_vec(),
                    shape: self.shape.clone(),
                });
            }
        }

        let offset: usize = idx
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();

        Ok(&self.data[offset])
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::matrics::Tensor;

    #[test]
    #[ignore]
    fn test_idx() {
        todo!("coming soon");
    }
}
