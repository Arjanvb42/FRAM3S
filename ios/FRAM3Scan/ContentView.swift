import ARKit
import SceneKit
import SwiftUI

/// Camera preview that shares its AR session with the scanner.
struct ARPreview: UIViewRepresentable {
    let scanner: Scanner

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.automaticallyUpdatesLighting = false
        scanner.attach(view.session)
        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var scanner = Scanner()
    private let teal = Color(red: 0, green: 0.70, blue: 0.64)

    var body: some View {
        ZStack {
            ARPreview(scanner: scanner).ignoresSafeArea()
            VStack(spacing: 12) {
                HStack {
                    Text("FRAM3").font(.headline).tracking(2)
                    Spacer()
                    Text(scanner.hasLidar ? "LiDAR aanwezig" : "Geen LiDAR")
                        .font(.footnote)
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .overlay(Capsule().stroke(scanner.hasLidar ? teal : Color.gray))
                        .foregroundColor(scanner.hasLidar ? teal : .gray)
                }
                .padding(12)
                .background(.black.opacity(0.55))
                .cornerRadius(12)

                if scanner.phase == .scanning {
                    HStack {
                        Text("\(scanner.sent)").font(.title.bold()).monospacedDigit()
                        Text("beelden")
                        Spacer()
                    }
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(.black.opacity(0.55))
                    .cornerRadius(10)
                }

                Spacer()

                Text(scanner.message)
                    .multilineTextAlignment(.center)
                    .padding(10)
                    .frame(maxWidth: .infinity)
                    .background(.black.opacity(0.55))
                    .cornerRadius(10)

                switch scanner.phase {
                case .pairing:
                    EmptyView()
                case .ready:
                    bigButton("Start") { scanner.start() }
                case .scanning:
                    bigButton(scanner.sent < Scanner.minFrames
                              ? "Nog minstens \(Scanner.minFrames - scanner.sent) beelden"
                              : "Klaar, maak het model") { scanner.finish() }
                        .disabled(scanner.sent < Scanner.minFrames)
                        .opacity(scanner.sent < Scanner.minFrames ? 0.6 : 1)
                case .done:
                    bigButton("Nieuwe scan") { scanner.reset() }
                }
            }
            .foregroundColor(.white)
            .padding(14)
        }
    }

    private func bigButton(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).fontWeight(.semibold).frame(maxWidth: .infinity).padding(15)
        }
        .background(teal)
        .cornerRadius(12)
    }
}
