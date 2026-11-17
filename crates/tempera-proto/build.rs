//! Compiles `proto/` with protox (pure Rust), so no `protoc` install is needed.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = "../../proto";
    let file = "tempera/agent/v1/agent.proto";
    println!("cargo:rerun-if-changed={root}/{file}");
    let fds = protox::compile([file], [root])?;
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_fds(fds)?;
    Ok(())
}
