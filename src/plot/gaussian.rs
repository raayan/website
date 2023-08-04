use num_traits::{FromPrimitive, Num, ToPrimitive};
use rand::prelude::Distribution;

use statrs::distribution::Continuous;
use std::marker::PhantomData;

pub trait MixtureModel<T: Distribution<V>, V> {
    fn observe(&mut self, point: T);
}

pub struct BoundedEstimator<'a, T: Distribution<K> + Continuous<&'a K, V>, V, K> {
    bounds: (&'a K, &'a K),
    pub observations: Vec<T>,
    pd_v: PhantomData<V>,
}

impl<'a, T: Distribution<K> + Continuous<&'a K, V>, V, K> BoundedEstimator<'a, T, V, K> {
    pub fn new(bounds: (&'a K, &'a K)) -> Self {
        Self {
            bounds,
            observations: vec![],
            pd_v: Default::default(),
        }
    }
}

impl<'a, T: Distribution<K> + Continuous<&'a K, V>, V, K> MixtureModel<T, K>
    for BoundedEstimator<'a, T, V, K>
{
    fn observe(&mut self, point: T) {
        self.observations.push(point);
    }
}

impl<'a, T: Distribution<K> + Continuous<&'a K, V>, V, K> Continuous<&'a K, V>
    for BoundedEstimator<'a, T, V, K>
where
    V: Copy + Default + Num + ToPrimitive + FromPrimitive,
    K: PartialOrd,
{
    fn pdf(&self, x: &'a K) -> V {
        if self.observations.is_empty() || ((x < self.bounds.0) && (x < self.bounds.1)) {
            V::default()
        } else {
            FromPrimitive::from_f64(
                (self
                    .observations
                    .iter()
                    .map(|o| o.pdf(x))
                    .fold(V::default(), |a, b| a + b)
                    .to_f64()
                    .unwrap())
                    * (1. / (self.observations.len() as f64)),
            )
            .unwrap()
        }
    }

    fn ln_pdf(&self, _x: &'a K) -> V {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use plotters::prelude::{IntoLinspace, Ranged};
    use statrs::distribution::{Continuous, Normal};

    #[test]
    fn test_normal_dist() {
        let dist = Normal::new(0.5, 0.5).unwrap();

        let resolution = 10;
        let points = (0..(resolution + 1))
            .step(1)
            .range()
            .map(|i| (i as f64) / (resolution as f64));

        let samples = points.map(|x| (x, dist.pdf(x))).collect::<Vec<_>>();
        println!("{:?}", samples);
    }
}
