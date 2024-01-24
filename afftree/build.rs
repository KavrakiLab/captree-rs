fn main() {
    cxx_build::bridge("src/lib.rs").compile("afftree");

    println!("cargo:rerun-if-changed=src/lib.rs");
}
