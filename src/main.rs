use itertools::Itertools;

fn train_hebb(inputs: &Vec<Vec<i32>>, outputs: &Vec<i32>, len_weights: usize) -> Vec<i32> {
    let mut new_weights = vec![0; len_weights];

    for (i, vetor) in inputs.iter().enumerate() {
        for (j, input) in vetor.iter().enumerate() {
            new_weights[j] += input * outputs[i];
        }
        new_weights[len_weights - 1] += outputs[i];
    }
    //println!("{new_weights:?}");

    new_weights
}

fn verify_truth_table(inputs: &Vec<Vec<i32>>, outputs: &Vec<i32>, weights: Vec<i32>) -> bool {
    for (i, vetor) in inputs.iter().enumerate() {
        let result = match vetor
            .iter()
            .enumerate()
            .map(|(j, input)| input * weights[j])
            .sum::<i32>()
            + weights.last().unwrap()  // bias
            >= 0
        {
            true => 1,
            false => -1,
        };

        if result != outputs[i] {
            return false;
        }

        //println!("{vetor:?} -> {result:?}");
    }

    true
}

fn main() {
    let inputs = vec![vec![-1, -1], vec![-1, 1], vec![1, -1], vec![1, 1]];

    let outputs = vec![-1, 1, -1, -1];
    let len_weights = 3;

    let weights = train_hebb(&inputs, &outputs, len_weights);
    let a = verify_truth_table(&inputs, &outputs, weights);
    println!("{a}");

    let test = vec![1, 1, 1, 1, -1, -1, -1, -1];
    
    let mut x = 0;
    let mut y = 0;

    for comb in test.iter().copied().permutations(4).unique() {
        println!("{comb:?}");
        x += 1;
        let weights = train_hebb(&inputs, &comb, 3);
        let a = verify_truth_table(&inputs, &comb, weights);
        if a {
            y += 1;
        }
    }
    println!("{y}/{x}");
}
