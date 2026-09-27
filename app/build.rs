//! On Windows, build the icon and the file's details into aegist-app.exe.
fn main() {
    println!("cargo:rerun-if-changed=aegist.rc");
    println!("cargo:rerun-if-changed=aegist.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile("aegist.rc", embed_resource::NONE).manifest_optional().expect("embedding the Windows icon");
    }
}
