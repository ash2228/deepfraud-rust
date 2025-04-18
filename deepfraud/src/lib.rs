use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

static LEARNING_RATE: f32 = 0.4;
static SEED: u64 = 123;

fn sigmoid_activation(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Serialize, Clone, Deserialize)]
struct Neuron {
    weight: Vec<f32>,
    bias: f32,
    output: f32,
    z: f32,
    prev_inputs: Vec<f32>,
}

struct NeuralLayer {
    neurons: Vec<Neuron>,
}

#[pyclass]
struct NeuralNetwork {
    inputs: Vec<Vec<f32>>,
    output: Vec<f32>,
    neural_layer: Vec<NeuralLayer>,
}

impl Neuron {
    fn new(input_size: i32) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(SEED);
        let weights: Vec<f32> = (0..input_size)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
        Neuron {
            weight: weights,
            bias: rng.random_range(0.0..1.0),
            output: 0.0,
            z: 0.0,
            prev_inputs: vec![],
        }
    }

    fn forward_pass(&mut self, inputs: Vec<f32>) -> f32 {
        self.prev_inputs = inputs.clone();
        self.z = self
            .weight
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum::<f32>()
            + self.bias;
        self.output = sigmoid_activation(self.z);
        self.output
    }

    fn backpropogation(&mut self, dl_doubt: f32) -> Vec<f32> {
        let sigmoid_derivative = self.output * (1.0 - self.output);
        let delta = dl_doubt * sigmoid_derivative;
        let dw: Vec<f32> = self.prev_inputs.iter().map(|inp| inp * delta).collect();
        let db = delta;

        self.weight = self
            .weight
            .iter()
            .zip(dw.iter())
            .map(|(w, d)| w - LEARNING_RATE * d)
            .collect();

        self.bias -= LEARNING_RATE * db;

        let mut output: Vec<f32> = vec![];
        for w in self.weight.iter() {
            output.push(w * delta);
        }
        output
    }
}

impl NeuralLayer {
    fn new(neuron_count: i32, input_size: i32) -> Self {
        let mut neurons: Vec<Neuron> = vec![];
        for _ in 0..neuron_count {
            neurons.push(Neuron::new(input_size));
        }
        NeuralLayer { neurons }
    }

    fn activate_layer(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        let mut outputs: Vec<f32> = vec![];
        for neuron in self.neurons.iter_mut() {
            outputs.push(neuron.forward_pass(inputs.clone()));
        }
        outputs
    }

    fn backward_pass(&mut self, grad: Vec<f32>) -> Vec<f32> {
        let mut new_grads: Vec<f32> = vec![0.0; self.neurons[0].prev_inputs.len()];
        for (neuron, gradi) in self.neurons.iter_mut().zip(grad.iter()) {
            let mut new_gradi = neuron.backpropogation(gradi.clone());
            for (a, b) in new_grads.iter_mut().zip(new_gradi.iter_mut()) {
                *a += *b;
            }
        }
        new_grads
    }
}

#[pymethods]
impl NeuralNetwork {
    #[new]
    fn new(inputs: Vec<Vec<f32>>, output: Vec<f32>, hidden_layers: Vec<i32>) -> Self {
        let mut neural_layer: Vec<NeuralLayer> = vec![];
        let mut input_size = inputs[0].len() as i32;
        for i in hidden_layers {
            neural_layer.push(NeuralLayer::new(i, input_size));
            input_size = i;
        }
        neural_layer.push(NeuralLayer::new(1, 5));
        NeuralNetwork {
            inputs,
            output,
            neural_layer,
        }
    }

    pub fn predict(&mut self, x: Vec<Vec<f32>>) -> Vec<f32> {
        x.into_iter()
            .map(|input| {
                let mut curr_input = input;
                for layer in self.neural_layer.iter_mut() {
                    curr_input = layer.activate_layer(curr_input);
                }
                curr_input[0]
            })
            .collect()
    }

    fn train(&mut self, iters: i32) {
        for _ in 0..iters {
            for (index, input) in self.inputs.iter_mut().enumerate() {
                let mut curr_input = input.clone();
                for layer in self.neural_layer.iter_mut() {
                    curr_input = layer.activate_layer(curr_input);
                }
                // let loss = (curr_input[0] - self.output[index]).powi(2);
                let mut grad = vec![2.0 * (curr_input[0] - self.output[index])];
                for layer in self.neural_layer.iter_mut().rev() {
                    grad = layer.backward_pass(grad.clone());
                }
            }
        }
    }

    pub fn write_to_json(&mut self, file_name: String) -> std::io::Result<()> {
        let mut data: Vec<Vec<Neuron>> = vec![];
        for neuron in self.neural_layer.iter_mut() {
            data.push(neuron.neurons.clone());
        }
        let json_string = serde_json::to_string_pretty(&data).unwrap();
        let mut file = File::create(format!("{}.json", file_name))?;
        file.write_all(json_string.as_bytes())?;

        Ok(())
    }

    pub fn load_config(&mut self, file_name: String) {
        // Read JSON file
        let file_path = format!("{}.json", file_name);
        let file_content = std::fs::read_to_string(file_path).expect("Failed to read config file");

        // Deserialize JSON to Vec<Vec<Neuron>>
        let config: Vec<Vec<Neuron>> =
            serde_json::from_str(&file_content).expect("Failed to parse config JSON");

        // Assign deserialized neurons to layers
        for (layer, new_neurons) in self.neural_layer.iter_mut().zip(config.into_iter()) {
            layer.neurons = new_neurons;
        }
    }
}

#[pymodule]
fn deepfraud(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuralNetwork>()?;
    Ok(())
}
