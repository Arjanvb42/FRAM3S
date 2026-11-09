import ARKit
import CoreImage
import SwiftUI
import UIKit
import Vision

/// Pairs with FRAM3 on the computer (QR code), then sends photos, camera
/// poses and LiDAR depth while you walk around the object.
final class Scanner: NSObject, ObservableObject, ARSessionDelegate, URLSessionDelegate {
    enum Phase { case pairing, ready, scanning, done }
    static let minFrames = 20

    @Published var phase: Phase = .pairing
    @Published var message = "Richt de camera op de QR-code in FRAM3 op je computer"
    @Published var sent = 0
    let hasLidar = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    private weak var session: ARSession?
    private let queue = DispatchQueue(label: "fram3.frames")
    private let ciContext = CIContext()
    private lazy var http = URLSession(configuration: .ephemeral, delegate: self, delegateQueue: nil)

    // Only touched on `queue`.
    private var base: URL?
    private var state: Phase = .pairing
    private var lastQRCheck = 0.0
    private var lastPose: simd_float4x4?
    private var inFlight = 0
    private var maxFrames = 400
    private var pairing = false

    // MARK: AR session

    func attach(_ session: ARSession) {
        self.session = session
        session.delegate = self
        session.delegateQueue = queue
        let config = ARWorldTrackingConfiguration()
        if hasLidar { config.frameSemantics.insert(.sceneDepth) }
        if let fourK = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution {
            config.videoFormat = fourK
        }
        session.run(config)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        switch state {
        case .pairing: lookForQR(frame)
        case .scanning: maybeCapture(frame)
        default: break
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        show("De camera kon niet starten: \(error.localizedDescription)")
    }

    // MARK: pairing

    private func lookForQR(_ frame: ARFrame) {
        guard !pairing, frame.timestamp - lastQRCheck > 0.3 else { return }
        lastQRCheck = frame.timestamp
        let request = VNDetectBarcodesRequest()
        request.symbologies = [.qr]
        let handler = VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, options: [:])
        try? handler.perform([request])
        for result in request.results ?? [] {
            guard let text = result.payloadStringValue, let url = URL(string: text),
                  url.scheme == "https", url.path.hasPrefix("/p/") else { continue }
            pair(url)
            return
        }
    }

    private func pair(_ url: URL) {
        pairing = true
        show("Verbinden met FRAM3…")
        base = url
        var name = UIDevice.current.model
        var info = utsname()
        uname(&info)
        let machine = withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        if !machine.isEmpty { name += " (\(machine))" }
        let hello: [String: Any] = ["name": name, "cameras": [], "lidar": hasLidar, "app": true,
                                    "width": 0, "height": 0]
        var request = URLRequest(url: url.appendingPathComponent("hello"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: hello)
        http.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }
            self.queue.async {
                self.pairing = false
                guard error == nil, (response as? HTTPURLResponse)?.statusCode == 200 else {
                    self.base = nil
                    self.show("Verbinden mislukt. Zit je telefoon op dezelfde wifi als je computer? Scan de code opnieuw.")
                    return
                }
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let limit = json["max_frames"] as? Int {
                    self.maxFrames = limit
                }
                self.set(.ready, "Verbonden. Zet het object op een vrije plek met goed licht en tik op Start.")
            }
        }.resume()
    }

    // MARK: scanning

    func start() {
        queue.async {
            self.lastPose = nil
            self.set(.scanning, "Loop rustig om het object heen: een rondje laag en een rondje hoog")
        }
    }

    func finish() {
        queue.async {
            guard self.state == .scanning, let base = self.base else { return }
            self.set(.done, "Klaar. FRAM3 maakt nu op je computer het 3D-model.")
            var request = URLRequest(url: base.appendingPathComponent("done"))
            request.httpMethod = "POST"
            self.http.dataTask(with: request).resume()
        }
    }

    func reset() {
        queue.async {
            self.base = nil
            DispatchQueue.main.async { self.sent = 0 }
            self.set(.pairing, "Richt de camera op de QR-code in FRAM3 op je computer")
        }
    }

    /// Take a frame when tracking is good and the phone has moved enough since the last one.
    private func maybeCapture(_ frame: ARFrame) {
        guard inFlight < 2, let base = base else { return }
        guard case .normal = frame.camera.trackingState else {
            show("Even rustig houden, de telefoon zoekt zijn positie")
            return
        }
        let pose = frame.camera.transform
        if let last = lastPose {
            let moved = simd_distance(pose.columns.3, last.columns.3)
            let turned = acos(min(1, max(-1, simd_dot(simd_normalize(pose.columns.2), simd_normalize(last.columns.2)))))
            if moved < 0.04 && turned < 0.10 { return }
        }
        guard let jpeg = jpegData(frame.capturedImage) else { return }
        lastPose = pose
        show("Goed zo, rustig doorlopen")

        let width = CVPixelBufferGetWidth(frame.capturedImage)
        let height = CVPixelBufferGetHeight(frame.capturedImage)
        let k = frame.camera.intrinsics
        var meta: [String: Any] = [
            "width": width, "height": height,
            // row-major, as the computer expects
            "intrinsics": (0..<3).flatMap { r in (0..<3).map { c in Double(k[c][r]) } },
            "pose": (0..<4).flatMap { r in (0..<4).map { c in Double(pose[c][r]) } },
        ]
        var parts: [(String, String, Data)] = [("image", "image/jpeg", jpeg)]
        if let depth = frame.sceneDepth {
            meta["depth_width"] = CVPixelBufferGetWidth(depth.depthMap)
            meta["depth_height"] = CVPixelBufferGetHeight(depth.depthMap)
            parts.append(("depth", "application/octet-stream", planeData(depth.depthMap, bytesPerPixel: 4)))
            if let confidence = depth.confidenceMap {
                parts.append(("conf", "application/octet-stream", planeData(confidence, bytesPerPixel: 1)))
            }
        }
        guard let metaData = try? JSONSerialization.data(withJSONObject: meta) else { return }

        let boundary = "fram3-\(UUID().uuidString)"
        var body = Data()
        func add(_ text: String) { body.append(text.data(using: .utf8)!) }
        add("--\(boundary)\r\nContent-Disposition: form-data; name=\"meta\"\r\n\r\n")
        body.append(metaData)
        add("\r\n")
        for (name, type, data) in parts {
            add("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"; filename=\"\(name)\"\r\nContent-Type: \(type)\r\n\r\n")
            body.append(data)
            add("\r\n")
        }
        add("--\(boundary)--\r\n")

        var request = URLRequest(url: base.appendingPathComponent("arframe"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        inFlight += 1
        http.uploadTask(with: request, from: body) { [weak self] data, response, error in
            guard let self = self else { return }
            self.queue.async {
                self.inFlight -= 1
                guard error == nil, let status = (response as? HTTPURLResponse)?.statusCode else {
                    self.show("Geen verbinding met FRAM3. Blijf op dezelfde wifi.")
                    return
                }
                if status == 200, let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let count = json["frames"] as? Int {
                    DispatchQueue.main.async { self.sent = count }
                    if count >= self.maxFrames { self.finish() }
                } else if status == 404 {
                    self.set(.pairing, "De koppeling is verlopen. Scan de QR-code in FRAM3 opnieuw.")
                } else if status == 409 {
                    self.finish()
                }
            }
        }.resume()
    }

    private func jpegData(_ buffer: CVPixelBuffer) -> Data? {
        let image = CIImage(cvPixelBuffer: buffer)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let quality = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        return ciContext.jpegRepresentation(of: image, colorSpace: space, options: [quality: 0.9])
    }

    /// Raw pixels of a one-plane buffer, row padding removed.
    private func planeData(_ buffer: CVPixelBuffer, bytesPerPixel: Int) -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var data = Data(capacity: width * height * bytesPerPixel)
        guard let start = CVPixelBufferGetBaseAddress(buffer) else { return data }
        for row in 0..<height {
            data.append(start.advanced(by: row * stride).assumingMemoryBound(to: UInt8.self),
                        count: width * bytesPerPixel)
        }
        return data
    }

    // MARK: helpers

    private func set(_ phase: Phase, _ text: String) {
        state = phase
        DispatchQueue.main.async {
            self.phase = phase
            self.message = text
        }
    }

    private func show(_ text: String) {
        DispatchQueue.main.async { if self.message != text { self.message = text } }
    }

    /// FRAM3 on the computer uses its own certificate inside your home network.
    /// Trust it for the paired address only (the one from the QR code).
    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        let space = challenge.protectionSpace
        var host: String?
        queue.sync { host = self.base?.host }
        if space.authenticationMethod == NSURLAuthenticationMethodServerTrust,
           let trust = space.serverTrust, let host = host, space.host == host {
            completionHandler(.useCredential, URLCredential(trust: trust))
        } else {
            completionHandler(.performDefaultHandling, nil)
        }
    }
}
