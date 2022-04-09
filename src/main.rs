use std::error::Error;
use rulinalg::matrix;
use rulinalg::matrix::Matrix;
use rand::thread_rng;
use rand::seq::SliceRandom;

type Record = (f32,f32,f32,f32,String);

fn run() -> Result<Vec<Record>, Box<dyn Error>> {
    let file_path = "/home/pepe/Rust_applications/neural-iris/iris-data-set.csv";
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut record : Record;
    let mut V : Vec<Record> = Vec::new();
    for result in rdr.deserialize() {
        record = result?;
        V.push(record);
    }
    Ok(V)
}

fn relu(a : f32) -> f32 {
    if a > 0.0 {return a;}
    a * 0.01
}

fn drelu(a : f32) -> f32 {
    if a > 0.0 {return 1.0;}
    0.01
}

fn relu_der(l : usize, m : Matrix<f32>) -> Matrix<f32> {
    let mut M = m;
    for i in 0..l {
        M[[0,i]] = drelu(M[[0,i]]);
    }
    M
}

enum Activ {
    ReLu,
    SoftMax,
}

struct Layer {
    inp : usize,
    out : usize,
    weights : Matrix<f32>,
    activ : Activ,
}

impl Layer {
    fn compute(&self, input : Matrix<f32>) -> Matrix<f32> {
        let mut output = input * self.weights.clone();
        match self.activ {
            Activ::ReLu => {
                for m in 0..self.out {
                    output[[0,m]] = relu(output[[0,m]]);
                }
            }
            Activ::SoftMax => {
                let mut exps = 0.0f32;
                for m in 0..self.out {
                    exps += f32::exp(output[[0,m]]);
                }
                for m in 0..self.out {
                    output[[0,m]] = f32::exp(output[[0,m]]) / exps;
                }
            }
        }
        output
    }
}

struct NN {
    Layers : Vec<Layer>,
}

fn rand_matrix(a : usize, b : usize) -> Matrix<f32> {
    Matrix::from_fn(a,b,|_,_| rand::random::<f32>())
}

fn trans(rows : usize, cols : usize, m : Matrix<f32>) -> Matrix<f32> {
    Matrix::from_fn(rows, cols, |col, row| m[[col,row]]) 
}

impl NN {
    fn data_train(&mut self, data : &Vec<Record>, rate : f32, momentum : f32, accuracy : f32) {
        let mut inputs : Vec<Matrix<f32>> = Vec::new();
        let mut outputs : Vec<Matrix<f32>> = Vec::new();
        for i in 0..data.len() {
            inputs.push(matrix!(data[i].0, data[i].1, data[i].2, data[i].3));
            outputs.push(to_output(&data[i].4.clone()[..]));
        }

        //повторяем структуру нейросети в этом блоке
        let mut ww : Vec<Matrix<f32>> = Vec::new();
        ww.push(Matrix::zeros(4,6));
        ww.push(Matrix::zeros(6,6));
        ww.push(Matrix::zeros(6,3));
        let mut acc = 1.0;
        while acc > accuracy {
            //Вычисляем выходы сети на определенных слоях для всех входных данных
            let mut inp;
            let mut X : Vec<Vec<Matrix<f32>>> = Vec::new();
            let mut x : Vec<Matrix<f32>>;
            
            for i in 0..data.len() {
                inp = inputs[i].clone();
                x = Vec::new();
                x.push(trans(4,1,inp.clone())); // вход
                inp = self.Layers[0].compute(inp);
                x.push(trans(6,1,inp.clone())); // выход 1 слоя 
                inp = self.Layers[1].compute(inp);
                x.push(trans(6,1,inp.clone())); // выход 2 слоя
                inp = self.Layers[2].compute(inp);
                x.push(trans(3,1,inp.clone())); // выход 3 слоя
                X.push(x);
            }

            //let mut W : Vec<Vec<Matrix<f32>>> = Vec::new();
            let mut WW : Vec<Matrix<f32>> = Vec::new();
            let mut w : Vec<Matrix<f32>>;
            let mut grad;
            WW.push(Matrix::zeros(4,6));
            WW.push(Matrix::zeros(6,6));
            WW.push(Matrix::zeros(6,3));

            for i in 0..data.len() {
                w = vec!(Matrix::zeros(0,0);3);
                //Считаем изначальный градиент
                grad = trans(1,3,&X[i][3] - trans(3,1,outputs[i].clone()));
                //Считаем градиенты матриц на промежуточных слоях
                w[2] = &X[i][2] * &grad;
                grad = grad * trans(3,6,self.Layers[2].weights.clone());
                w[1] = &X[i][1] * &grad;
                grad = relu_der(6,grad);
                grad = grad * trans(6,6,self.Layers[1].weights.clone());
                w[0] = &X[i][0] * &grad;
                //W.push(w.clone());
                for k in 0..3 {
                    WW[k] += &w[k];
                }
            }

            for k in 0..3 {
                WW[k] /= data.len() as f32;
            }
            //Применяем градиенты для рассчета новых весов в сети

            for j in 0..3 {
                self.Layers[j].weights -= &WW[j] * rate + &ww[j] * momentum;
            }
            ww = WW;
            
            acc = 0.0;
            for i in 0..data.len() {
                let res = self.compute(inputs[i].clone());
                for r in 0..3 {
                    acc += (res[[0,r]] - outputs[i][[0,r]]).powi(2);
                }
            }
            acc /= data.len() as f32;
            println!("{}",acc);
        }
    }
    fn compute(&self, input : Matrix<f32>) -> Matrix<f32> {
        let mut output = input;
        for l in 0..self.Layers.len() {
            output = self.Layers[l].compute(output);
        }
        output
    }
    fn new() -> NN {
        let mut N = NN{Layers : Vec::new()};
        N.Layers.push(
            Layer{
                inp : 4,
                out : 6,
                weights : rand_matrix(4,6), 
                activ : Activ::ReLu
            }
        );
        N.Layers.push(
            Layer{
                inp : 6,
                out : 6,
                weights : rand_matrix(6,6), 
                activ : Activ::ReLu
            }
        );
        N.Layers.push(
            Layer{
                inp : 6,
                out : 3,
                weights : rand_matrix(6,3), 
                activ : Activ::SoftMax
            }
        );
        N
    }
}

fn to_output(s : &str) -> Matrix<f32> {
    match s {
        "setosa"     => {return matrix!(1.0, 0.0, 0.0);}
        "versicolor" => {return matrix!(0.0, 1.0, 0.0);}
        "virginica"  => {return matrix!(0.0, 0.0, 1.0);}
        _ => {panic!("no such flower type");}
    }
}

fn to_ans(m : Matrix<f32>) -> String {
    if m[[0,0]] >= m[[0,1]] && m[[0,0]] >= m[[0,2]] {return String::from("setosa");}
    if m[[0,1]] >= m[[0,0]] && m[[0,1]] >= m[[0,2]] {return String::from("versicolor");}
    String::from("virginica")
}

fn main() {
    let cont = run();
    let mut V : Vec<Record> = Vec::new();
    match cont {
        Ok(v) => {
            V = v;
        }
        Err(err) => {println!("{}",err);}
    }

    let mut Netw = NN::new();
    /*let inp = matrix!(1.0, 1.0, 1.0, 1.0);
    let out = Netw.compute(inp.clone());
    println!("{}",out);*/
    let mut pos : u32 = 0;
    V.shuffle(&mut thread_rng());
    let mut W : Vec<Record> = Vec::new();
    for v in &V {
        W.push(v.clone());
    }
    Netw.data_train(&V,0.006,0.003,0.0005);
    
    for o in 100..150 {
        println!("{}: '{}'",o,V[o].4);

        let input  = matrix!(V[o].0, V[o].1, V[o].2, V[o].3);
        let out = Netw.compute(input);
        
        println!("{}",out.clone());
        println!("{}",to_output(&V[o].4[..]));
        println!();
        
        if V[o].4 == to_ans(out.clone()) {pos+=1;}
    }
    println!("{}%",pos as f32 / 0.5);
}
