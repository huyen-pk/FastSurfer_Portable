use super::driver::DriverFrame;
use super::message::TransportMessage;

pub trait TransportProtocol: Send + Sync + 'static {
    /// Encodes a transport request into a driver frame.
    ///
    /// # Errors
    /// Returns an error when the message cannot be represented by the protocol.
    fn encode_request(
        &self,
        message: &TransportMessage,
    ) -> Result<DriverFrame, String>;

    /// Decodes a driver frame into one or more transport messages.
    ///
    /// # Errors
    /// Returns an error when the inbound frame violates protocol expectations.
    fn decode_inbound(
        &self,
        frame: DriverFrame,
    ) -> Result<Vec<TransportMessage>, String>;
}
