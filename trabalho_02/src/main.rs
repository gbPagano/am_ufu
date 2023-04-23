use std::io;


fn verify_xor(w: i32, teta: i32) {
    
    let a = ((1 * w + 1 * w) >= teta) as i8;
    let b = ((0 * w + 1 * w) >= teta) as i8;
    let c = ((1 * w + 0 * w) >= teta) as i8;
    let d = ((0 * w + 0 * w) >= teta) as i8;

    println!("\nTabela verdade");
    println!("--------------");
    println!("1 | 1 -> {a}");
    println!("0 | 1 -> {b}");
    println!("1 | 0 -> {c}");
    println!("0 | 0 -> {d}");


}


fn main() {
    println!("insira o valor do peso e do teta: ");

    let mut peso = String::new();
    io::stdin()
        .read_line(&mut peso)
        .expect("Failed to read line");
    let peso: i32 = peso.trim().parse().expect("Please type a number!");


    let mut teta = String::new();
    io::stdin()
        .read_line(&mut teta)
        .expect("Failed to read line");
    let teta: i32 = teta.trim().parse().expect("Please type a number!");

    verify_xor(peso, teta);
}
