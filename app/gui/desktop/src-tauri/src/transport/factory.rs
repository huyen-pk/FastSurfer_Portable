use super::Transport;
use super::ipc_protocol::FastSurferJsonLineProtocol;
use super::protocol::TransportProtocol;
use super::stdio::StdioTransportDriver;
use crate::backend::subprocess::BackendProcess;
use std::sync::Arc;

/// Creates the current `stdio + FastSurfer JSON-line` transport pair.
///
/// # Errors
/// Returns an error when the process streams cannot be wrapped in the
/// configured transport runtime.
pub fn create_fastsurfer_stdio_transport(
    process: BackendProcess,
) -> Result<(std::process::Child, Transport), String> {
    let (child, stdin, stdout) = process.into_parts();
    let protocol: Arc<dyn TransportProtocol> =
        Arc::new(FastSurferJsonLineProtocol::new());
    let driver = Box::new(StdioTransportDriver::new(stdin, stdout));
    let transport = Transport::new(driver, protocol)?;
    Ok((child, transport))
}

/// Creates the current `stdio + FastSurfer JSON-line` transport pair from owned stdio parts.
///
/// # Errors
/// Returns an error when the stdio streams cannot be wrapped in the configured
/// transport runtime.
pub fn create_fastsurfer_stdio_transport_from_parts(
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
) -> Result<Transport, String> {
    let protocol: Arc<dyn TransportProtocol> =
        Arc::new(FastSurferJsonLineProtocol::new());
    let driver = Box::new(StdioTransportDriver::new(stdin, stdout));
    Transport::new(driver, protocol)
}
