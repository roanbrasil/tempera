//! Rebuild when the web UI build output changes, so rust-embed picks it up.
fn main() {
    println!("cargo:rerun-if-changed=../../ui/build");
    println!("cargo:rerun-if-changed=../../ui/build/index.html");
}
