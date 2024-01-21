use std::io;
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

fn main() {
    let model_options = ModelOptions {
        n_gpu_layers: 100,
        context_size: 1024,
        // n_batch: 512,
        // embeddings: true,
        ..Default::default()
    };

    println!("{:?}", model_options);

    let llama = LLama::new(
        "./models/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF/dolphin-2.5-mixtral-8x7b.Q4_K_M.gguf".into(), 
        &model_options,
    )
    .expect("Could not load the model.");

    loop {
        println!(">>> ");
        let prompt = read_line();

        let text = format!("Below is an instruction that describes a task. Write a response that appropriately completes that request.\n{}",
                            prompt);

        let predict_options = PredictOptions {
            tokens: 400,
            token_callback: Some(Box::new(|token| {
                print!("{}", token);
                true
            })),
            ..Default::default()
        };

        llama
            .predict(
                text,
                predict_options,
            )
            .unwrap();

        println!("");
    }
}

fn read_line() -> String {
     // Create a new, empty String
     let mut input = String::new();
 
     // Read a line of input from the user and store it in the 'input' variable
     match io::stdin().read_line(&mut input) {
         Ok(_) => {
             println!("You typed: {}", input.trim());
         }
         Err(error) => {
             println!("Error: {}", error);
         }
     }

     input
}