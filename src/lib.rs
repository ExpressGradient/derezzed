use ndarray::{concatenate, Array1, Array2, Axis};
use std::ops::SubAssign;

pub struct LinearRegression {
    x: Array2<f64>,
    y: Array1<f64>,
    theta: Array1<f64>,
    iterations: i64,
    learning_rate: f64,
}

impl LinearRegression {
    pub fn new(x: Array2<f64>, y: Array1<f64>) -> Self {
        let n = x.shape()[1];

        Self {
            x: Self::with_bias(x),
            y,
            theta: Array1::zeros(n + 1),
            iterations: 1000,
            learning_rate: 0.01,
        }
    }

    fn with_bias(x: Array2<f64>) -> Array2<f64> {
        concatenate![Axis(1), Array2::ones((x.shape()[0], 1)), x]
    }

    pub fn fit(&mut self) {
        for _ in 0..self.iterations {
            let predictions = self.x.dot(&self.theta);

            let errors = predictions - &self.y;

            let step = self.x.t().dot(&errors) * self.learning_rate / self.x.shape()[0] as f64;
            self.theta.sub_assign(&step);
        }
    }

    pub fn predict(&self, x: Array2<f64>) -> Array1<f64> {
        let x_with_bias = Self::with_bias(x);

        x_with_bias.dot(&self.theta)
    }
}

#[cfg(test)]
mod tests {
    use super::LinearRegression;
    use ndarray::Array;

    #[test]
    fn test_linear_regression() {
        let x = Array::from_shape_vec((6, 2), vec![2., 1., 3., 1., 4., 1., 2., 2., 3., 2., 3., 3.])
            .unwrap();
        let y = Array::from_shape_vec((6,), vec![2., 4., 6., 2.5, 4.5, 5.]).unwrap();

        let mut lr = LinearRegression::new(x, y);
        lr.iterations = 100;

        lr.fit();

        let test_x = Array::from_shape_vec((1, 2), vec![3., 2.]).unwrap();
        let prediction = lr.predict(test_x);

        println!("Prediction: {}", prediction);
        assert!(prediction.get(0).unwrap() > &4.2 && prediction.get(0).unwrap() < &4.8);
    }
}
