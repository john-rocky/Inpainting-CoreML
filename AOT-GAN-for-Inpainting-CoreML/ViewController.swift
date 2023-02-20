import UIKit
import Vision
import CoreML
import Drawsana
import PhotosUI

class ViewController: UIViewController,PHPickerViewControllerDelegate, UIPickerViewDelegate {
    
    var imageView:UIImageView!
    let drawingView = DrawsanaView()
    let penTool = PenTool()
    let undoButton = UIButton()
    let runButton = UIButton()
    let selectPhotoButton =  UIButton()
    let superResolutionButton = UIButton()
    let brushButton = UIButton()
    var inputImage: UIImage?
    let ciContext = CIContext()
    lazy var maskBackGroundImage: UIImage = {
        guard let image = UIImage(named: "maskBackGround") else { fatalError("Please set black image ") }
        return image
    }()
    
    lazy var model: MLModel? = {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU
            let model = try aotgan(configuration: config).model
            return model
        } catch let error {
            print(error)
            fatalError("model initialize error")
        }
    }()
    
    lazy var srRequest: VNCoreMLRequest = {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU
            let model = try realesrgangeneral512(configuration: config).model
            let vnModel = try VNCoreMLModel(for: model)
            let request = VNCoreMLRequest(model: vnModel)
            request.imageCropAndScaleOption = .scaleFill
            return request
        } catch let error {
            print(error)
            fatalError("model initialize error")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupView()
    }
    
    
    func inference(maskedImage inputImage:UIImage, maskImage mask:UIImage) {
        guard let model = model else { fatalError("Model initialize error.") }
        do {
            let mask_1 = mask.mlMultiArrayGrayScale(scale: 1/255)

            // input
            
            let x_1 = inputImage.mlMultiArray(scale: 1/127.5,rBias: -1,gBias: -1, bBias: -1)
            
            let inputTensor = MLMultiArray(concatenating: [x_1, mask_1],
                                           axis: 1,
                                           dataType: .float32)
            
            let input = aotganInput(x_1: inputTensor)
            
            // run
            let start = Date()
            let out = try model.prediction(from: input)
            let timeElapsed = -start.timeIntervalSinceNow
            print(timeElapsed)
            let outArray = out.featureValue(for: "var_915")?.multiArrayValue
            let outImage = outArray!.cgImage(min: -1,max: 1, axes: (1,2,3))
            let uiOut = UIImage(cgImage: outImage!)
            let originalSize = self.inputImage?.size

            let comp = uiOut.mlMultiArrayComposite(outImage: uiOut, inputImage: inputImage, maskImage: mask).cgImage(axes: (1,2,3))?.resize(size: originalSize!)
            let final = UIImage(cgImage: comp!)
            
            self.inputImage = final
            
            DispatchQueue.main.async {

                self.resetDrawingView()
                self.imageView.image = final
            }
            print("Done")
        } catch let error {
            print(error)
        }
    }
    
    func resetDrawingView() {
        if drawingView.drawing.shapes.count != 0 {
            for _ in 0...drawingView.drawing.shapes.count-1 {
                drawingView.operationStack.undo()
            }
        }
    }
    
    func setupView() {
        imageView = UIImageView(frame: view.bounds)
        imageView.contentMode = .scaleAspectFit
        view.addSubview(imageView)
        inputImage = UIImage(named: "input")
        imageView.image = inputImage
        
        drawingView.set(tool: penTool)
        drawingView.userSettings.strokeWidth = 20
        drawingView.userSettings.strokeColor = .white
        drawingView.userSettings.fillColor = .black
        
        drawingView.userSettings.fontSize = 24
        drawingView.userSettings.fontName = "Marker Felt"
        drawingView.frame = imageView.frame
        view.addSubview(drawingView)
        let buttonWidth = view.bounds.width*0.3
        undoButton.frame = CGRect(x: view.bounds.width*0.025, y: 50, width: buttonWidth, height: 50)
        superResolutionButton.frame = CGRect(x: view.bounds.maxX-view.bounds.width*0.025-buttonWidth, y: 50, width: buttonWidth, height: 50)
        brushButton.frame = CGRect(x: undoButton.frame.maxX + view.bounds.width * 0.025, y: 50, width: buttonWidth, height: 50)
        selectPhotoButton.frame = CGRect(x: view.bounds.width*0.1, y:  view.bounds.maxY - 100, width: buttonWidth, height: 50)
        runButton.frame = CGRect(x: view.bounds.maxX - view.bounds.width*0.1 - buttonWidth, y: view.bounds.maxY - 100, width: buttonWidth, height: 50)
        undoButton.setTitle("undoDraw", for: .normal)
        selectPhotoButton.setTitle("select Photo", for: .normal)
        superResolutionButton.setTitle("SR", for: .normal)
        brushButton.setTitle("brushWidth", for: .normal)
        runButton.setTitle("run", for: .normal)
        undoButton.backgroundColor = .gray
        undoButton.setTitleColor(.white, for: .normal)
        selectPhotoButton.backgroundColor = .gray
        selectPhotoButton.setTitleColor(.white, for: .normal)
        superResolutionButton.backgroundColor = .gray
        superResolutionButton.setTitleColor(.white, for: .normal)
        brushButton.backgroundColor = .gray
        brushButton.setTitleColor(.white, for: .normal)
        runButton.backgroundColor = .gray
        runButton.setTitleColor(.white, for: .normal)
        selectPhotoButton.addTarget(self, action: #selector(presentPhPicker), for: .touchUpInside)
        superResolutionButton.addTarget(self, action: #selector(sr), for: .touchUpInside)
        brushButton.addTarget(self, action: #selector(brushWidth), for: .touchUpInside)

        undoButton.addTarget(self, action: #selector(undo), for: .touchUpInside)
        runButton.addTarget(self, action: #selector(run), for: .touchUpInside)
        view.addSubview(selectPhotoButton)
        view.addSubview(superResolutionButton)
        view.addSubview(undoButton)
        view.addSubview(brushButton)

        view.addSubview(runButton)
        adjustDrawingViewSize()
    }
    
    @objc func brushWidth(){
        let ac = UIAlertController(title: "SelectBrush", message: "", preferredStyle: .alert)
        ac.addAction(UIAlertAction(title: "thin", style: .default,handler: { action in
            self.drawingView.userSettings.strokeWidth = 5
        }))
        ac.addAction(UIAlertAction(title: "medium", style: .default,handler: { action in
            self.drawingView.userSettings.strokeWidth = 20
        }))
        ac.addAction(UIAlertAction(title: "thin", style: .default,handler: { action in
            self.drawingView.userSettings.strokeWidth = 40
        }))
        present(ac, animated: true)
    }
    
    @objc func undo() {
        drawingView.operationStack.undo()
    }
    
    @objc func run() {
        guard let overlapImage:UIImage = drawingView.render(over:inputImage),
              let maskImage:UIImage = drawingView.render()
        else { fatalError("Mask overlap error") }
        inference(maskedImage: overlapImage, maskImage: maskImage)
    }
    
    @objc func presentPhPicker(){
        var configuration = PHPickerConfiguration()
        configuration.selectionLimit = 1
        configuration.filter = .images
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = self
        present(picker, animated: true)
    }
    
    @objc func sr() {
        let handler = VNImageRequestHandler(ciImage: CIImage(image: inputImage!)!)
        do {
            try handler.perform([srRequest])
            guard let result = srRequest.results?.first as? VNPixelBufferObservation else {
                return
            }
            let srCIImage = CIImage(cvPixelBuffer: result.pixelBuffer)
            let resizedCGImage = ciContext.createCGImage(srCIImage, from: srCIImage.extent)?.resize(size: CGSize(width: inputImage!.size.width, height: inputImage!.size.height))
            let srUIImage = UIImage(cgImage: resizedCGImage!)
            inputImage = srUIImage
            DispatchQueue.main.async {
                self.imageView.image = srUIImage
                self.adjustDrawingViewSize()
            }
        } catch let error {
            print(error)
        }
    }
    
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        guard let result = results.first else { return }
        if result.itemProvider.canLoadObject(ofClass: UIImage.self) {
            result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] image, error  in
                if let image = image as? UIImage,  let safeSelf = self {
                    let correctOrientImage = safeSelf.getCorrectOrientationUIImage(uiImage: image)
                    safeSelf.inputImage = correctOrientImage
                    DispatchQueue.main.async {
                        safeSelf.resetDrawingView()
                        safeSelf.imageView.image = correctOrientImage
                        safeSelf.adjustDrawingViewSize()
                    }
                }
            }
        }
    }
    
    func getCorrectOrientationUIImage(uiImage:UIImage) -> UIImage {
        var newImage = UIImage()
        let ciContext = CIContext()
        switch uiImage.imageOrientation.rawValue {
        case 1:
            guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.down),
                  let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent) else { return uiImage}
            
            newImage = UIImage(cgImage: cgImage)
        case 3:
            guard let orientedCIImage = CIImage(image: uiImage)?.oriented(CGImagePropertyOrientation.right),
                  let cgImage = ciContext.createCGImage(orientedCIImage, from: orientedCIImage.extent) else { return uiImage}
            newImage = UIImage(cgImage: cgImage)
        default:
            newImage = uiImage
        }
        return newImage
    }
    
    func adjustDrawingViewSize() {
        let displayAspect = imageView.frame.height / imageView.frame.width
        let imageSize = imageView.image!.size
        let imageAspect = imageSize.height / imageSize.width
        if imageAspect <= displayAspect {
            // fit to width
            let minX = imageView.frame.minX
            let minY = imageView.center.y - (imageView.frame.width * imageAspect / 2)
            let width = imageView.frame.width
            let height = imageView.frame.width * imageAspect
            drawingView.frame = CGRect(x: minX, y: minY, width: width, height: height)

        } else {
            // fit to height
            let aspect = imageSize.width / imageSize.height
            drawingView.frame = CGRect(x: imageView.center.x - (imageView.frame.height * aspect / 2), y: imageView.frame.minY, width: imageView.frame.height * aspect, height: imageView.frame.height)
        }
        
    }
}

