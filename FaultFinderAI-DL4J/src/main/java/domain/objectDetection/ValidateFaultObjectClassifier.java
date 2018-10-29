package domain.objectDetection;

import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.jlab.groot.base.ColorPalette;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import faultrecordreader.FaultObjectDetectionRecordReader;
import faultrecordreader.FaultRecorderScaler;
import faults.FaultNames;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class ValidateFaultObjectClassifier {
	private String fileName;
	private DataSetIterator test;
	private FaultObjectClassifier classifier;
	private FaultObjectDetectionRecordReader recordReader;

	public ValidateFaultObjectClassifier(String fileName) {
		this.fileName = fileName;
		initialize();
		try {
			loadClassifier();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			throw new RuntimeException("Couldn't load classifier. Path might be incorrect.", e);
		}
	}

	private void initialize() {
		this.recordReader = new FaultObjectDetectionRecordReader(3, 10, FaultNames.CHANNEL_ONE, true, true, 112, 6, 1,
				56, 3);
		FaultRecordScalerStrategy strategy = new MinMaxStrategy();
		this.test = new RecordReaderDataSetIterator.Builder(recordReader, 1).regression(1).maxNumBatches(1)
				.preProcessor(new FaultRecorderScaler(strategy)).build();
	}

	private void loadClassifier() throws IOException {
		this.classifier = new FaultObjectClassifier(fileName);
	}

	public FaultObjectClassifier getClassifier() {
		return this.classifier;
	}

	public DataSetIterator getDSIterator() {
		return this.test;
	}

	public FaultObjectDetectionRecordReader getRecordReader() {
		return this.recordReader;
	}

	public void loadImage(INDArray arr) throws java.lang.Exception {
		// INDArray arr = recordReader.getFactory().getFeatureVectorAsMatrix();
		double max = (double) arr.maxNumber();

		System.out.println(arr.size(0) + "  " + arr.size(1) + "   max " + max);
		int xLength = arr.columns();
		int yLength = arr.rows();
		System.out.println(xLength + "  " + yLength);

		// BufferedImage b = new BufferedImage(yLength, xLength, 3);
		BufferedImage b = new BufferedImage(yLength, xLength, BufferedImage.TYPE_INT_RGB);
		ColorPalette palette = new ColorPalette();
		for (int y = 0; y < yLength; y++) {
			for (int x = 0; x < xLength; x++) {

				Color weightColor = palette.getColor3D(arr.getDouble(y, x), max, false);

				int red = weightColor.getRed();
				int green = weightColor.getGreen();
				int blue = weightColor.getBlue();

				// int rgb = weightColor.getRGB();
				int rgbII = (red * 65536) + (green * 256) + blue;
				// System.out.println(rgbII + " " + arr.getDouble(y, x) + " " +
				// weightColor.toString());
				b.setRGB(y, xLength - x - 1, rgbII);

			}
		}
		CanvasFrame cframe = new CanvasFrame("test");
		cframe.setTitle("test - HouseNumberDetection");
		cframe.setCanvasSize(800, 600);
		cframe.showImage(b);
		// cframe.waitKey();

		try {
			ImageIO.write(b, "png", new File("/Users/Mike/Desktop/ScreenShots/Doublearray.png"));

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("end");
		recordReader.getFactory().draw();
	}

	public static void main(String[] args) {
		int gridWidth = 6;
		int gridHeight = 112;

		FaultObjectClassifier classifier;
		String fileName = "models/binary_classifiers/ComputationalGraphModel/TestMKI.zip";
		// List<String> labels = train.getLabels();
		ValidateFaultObjectClassifier vObjectClassifier = new ValidateFaultObjectClassifier(fileName);

		ComputationGraph model = vObjectClassifier.getClassifier().getModel();
		org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model
				.getOutputLayer(0);
		RecordReaderDataSetIterator test = (RecordReaderDataSetIterator) vObjectClassifier.getDSIterator();
		org.nd4j.linalg.dataset.DataSet ds = test.next();
		List<String> labels = test.getLabels();
		System.out.println(ds.getFeatureMatrix().columns() + "   " + ds.getFeatureMatrix().rows());

		try {
			vObjectClassifier.loadImage(vObjectClassifier.getRecordReader().getFactory().getFeatureVectorAsMatrix());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// for (String string : labels) {
		// System.out.println(string);
		// }
		// RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)
		// ds.getExampleMetaData().get(0);
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

		INDArray features = ds.getFeatures();
		INDArray results = model.outputSingle(features);
		List<DetectedObject> objs = yout.getPredictedObjects(results, 0.0001);

		NativeImageLoader imageLoader = new NativeImageLoader();
		Mat mat = imageLoader.asMat(features);
		Mat convertedMat = new Mat();
		mat.convertTo(convertedMat, CV_8U, 255, 0);
		int w = 6 * 2;
		int h = 112 * 2;
		Mat image = new Mat();

		resize(convertedMat, image, new Size(w, h));
		System.out.println(objs.size() + "  size of objs");
		for (DetectedObject obj : objs) {
			System.out.println(obj.toString());
			double[] xy1 = obj.getTopLeftXY();
			double[] xy2 = obj.getBottomRightXY();
			String label = labels.get(obj.getPredictedClass());
			int x1 = (int) Math.round(w * xy1[0] / gridWidth);
			int y1 = (int) Math.round(h * xy1[1] / gridHeight);
			int x2 = (int) Math.round(w * xy2[0] / gridWidth);
			int y2 = (int) Math.round(h * xy2[1] / gridHeight);
			rectangle(image, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
			putText(image, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
		}
		CanvasFrame frame = new CanvasFrame("HouseNumberDetection");
		frame.setTitle(" - HouseNumberDetection");
		frame.setCanvasSize(w, h);
		frame.showImage(converter.convert(image));
		try {
			frame.waitKey();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
