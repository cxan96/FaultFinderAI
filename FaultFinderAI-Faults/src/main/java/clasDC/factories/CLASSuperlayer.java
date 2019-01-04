/**
 * 
 */
package clasDC.factories;

import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;

import clasDC.faults.Fault;
import clasDC.faults.FaultCoordinates;
import clasDC.faults.FaultNames;
import clasDC.faults.FaultSetter;
import lombok.Builder;
import utils.FaultUtils;

/**
 * @author m.c.kunkel <br>
 *         a CLASSuperlayer contains 6 layers and 112 wires. <br>
 *         FaultFactory provides the layers and wire faults.
 */

public class CLASSuperlayer extends CLASComponent {

	private AbstractFaultFactory factory = null;
	private int superlayer;

	@Builder
	private CLASSuperlayer(int superlayer, int nchannels, int minFaults, int maxFaults, FaultNames desiredFault,
			double desiredFaultGenRate, List<FaultNames> desiredFaults, boolean singleFaultGen, boolean isScaled) {
		if (superlayer > 6 || superlayer < 1) {
			throw new IllegalArgumentException("Invalid input: (superlayer), must have values less than"
					+ " ( 7) and more than (0). Received: (" + superlayer + ")");
		}
		this.superlayer = superlayer;
		this.nchannels = nchannels;
		this.minFaults = minFaults;
		this.maxFaults = maxFaults;
		this.desiredFault = desiredFault;
		this.desiredFaultGenRate = desiredFaultGenRate;
		this.desiredFaults = desiredFaults;
		this.singleFaultGen = singleFaultGen;
		this.isScaled = isScaled;
		this.factory = SingleFaultFactory.builder().superLayer(superlayer).minFaults(this.minFaults)
				.maxFaults(maxFaults).desiredFault(desiredFault).desiredFaults(desiredFaults).randomSuperlayer(true)
				.randomSmear(true).nChannels(nchannels).singleFaultGen(singleFaultGen)
				.desiredFaultGenRate(desiredFaultGenRate).build();
		init();

	}

	private void init() {
		if (this.isScaled) {
			this.image = FaultUtils.asImage(this.nchannels,
					FaultUtils.scaleImage(factory.asImageMatrix(1).getImage(), 6, 112));
			scaleFaults(factory.getFaultList());
			this.segmentationLabels = factory.getLabels();

		} else {
			this.image = factory.asUnShapedImageMatrix();
			this.faultList = factory.getFaultList();
			this.segmentationLabels = factory.getLabels();
		}

	}

	private void scaleFaults(List<Fault> faults) {
		List<Fault> aFaults = new ArrayList<>();
		for (Fault fault : faults) {
			fault.scaleFaultCoodinates(this);
			aFaults.add(fault);
		}
		this.faultList = aFaults;

	}

	public CLASSuperlayer getNewSuperLayer(int superLayer) {
		CLASSuperlayer sl = CLASSuperlayer.builder().superlayer(superLayer).nchannels(this.nchannels)
				.minFaults(this.minFaults).maxFaults(this.maxFaults).desiredFault(this.desiredFault)
				.desiredFaults(this.desiredFaults).singleFaultGen(this.singleFaultGen).isScaled(this.isScaled)
				.desiredFaultGenRate(this.desiredFaultGenRate).build();
		return sl;

	}

	public CLASFactory getNewFactory() {
		CLASFactory sl = CLASSuperlayer.builder().superlayer(this.superlayer).nchannels(this.nchannels)
				.minFaults(this.minFaults).maxFaults(this.maxFaults).desiredFault(this.desiredFault)
				.desiredFaults(this.desiredFaults).singleFaultGen(this.singleFaultGen).isScaled(this.isScaled)
				.desiredFaultGenRate(this.desiredFaultGenRate).build();
		return sl;

	}

	public Map<String, INDArray> locationLabels() {
		return factory.locationLabels();
	}

	public Map<FaultNames, INDArray> faultLocationLabels() {
		return factory.faultLocationLabels();
	}

	protected void testDrawFaults(CLASSuperlayer sl) {
		int w = 416;// 112 * 4;
		int h = 416;// 6 * 100;
		int gridWidth = 416;
		int gridHeight = 416;

		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

		NativeImageLoader imageLoader = new NativeImageLoader();
		INDArray a = sl.getImage().getImage();
		// INDArray b = a.reshape(ArrayUtil.combine(new long[] { 1 },
		// a.shape()));
		// b = FaultUtils.scaleImage(b, 6, 112);

		FaultUtils.draw(a);
		// FaultUtils.draw(sl.getSegmentationLabels());
		// FaultUtils.draw(b);
		Mat mat = imageLoader.asMat(a, 0);
		// Mat mat =
		// imageLoader.asMat(FaultUtils.zeroBorder(a));

		Mat convertedMat = new Mat();
		// mat.convertTo(convertedMat, CV_8U, 1, 0);
		mat.convertTo(convertedMat, 0, 1, 0);

		Mat image = new Mat();

		resize(convertedMat, image, new Size(w, h));

		for (Fault fault : sl.getFaultList()) {
			fault.printWireInformation();
			FaultCoordinates c = fault.getFaultCoordinates();
			double[] xy1 = new double[] { c.getXCenterPixels() - (c.getXMax() - c.getXMin()) / 2.0,
					c.getYCenterPixels() - (c.getYMax() - c.getYMin()) / 2.0 };
			double[] xy2 = new double[] { c.getXCenterPixels() + (c.getXMax() - c.getXMin()) / 2.0,
					c.getYCenterPixels() + (c.getYMax() - c.getYMin()) / 2.0 };
			String label = fault.getSubFaultName().getFamilyOf();
			int x1 = (int) Math.round((double) w * (c.getXMin()) / (double) gridWidth);
			int y1 = (int) Math.round((double) h * (c.getYMin()) / (double) gridHeight);
			int x2 = (int) Math.round((double) w * ((c.getXMax())) / (double) gridWidth);
			int y2 = (int) Math.round((double) h * ((c.getYMax())) / (double) gridHeight);
			int centerX = (int) Math.round((double) h * (c.getXCenterPixels()) / (double) gridHeight);
			int centerY = (int) Math.round((double) h * (c.getYCenterPixels()) / (double) gridHeight);

			System.out.println(x1 + "  " + "  " + y1 + "  " + x2 + "  " + y2);
			rectangle(image, new Point(x1, y1), new Point(x2, y2), Scalar.YELLOW);
			rectangle(image, new Point(centerX, centerY), new Point(centerX, centerY), Scalar.GREEN);

			putText(image, label, new Point(x1, ThreadLocalRandom.current().nextInt(0, 2) == 1 ? centerY : y2),
					FONT_HERSHEY_DUPLEX, 0.5, Scalar.GREEN);
		}
		CanvasFrame frame = new CanvasFrame("Valididate");
		frame.setTitle(" Fault - Valididation");
		frame.setCanvasSize(w, h);
		frame.showImage(converter.convert(image));
		// try {
		// frame.waitKey();
		// } catch (InterruptedException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
	}

	protected void testDrawFaults() {

		int[] preferredImageSize = { 448, 450 };

		INDArray before = this.getImage().getImage().dup();
		before = before.reshape(ArrayUtil.combine(new long[] { 1 }, before.shape()));
		INDArray a = FaultUtils.toColor(3, FaultUtils.scaleImage(before, preferredImageSize,
				FaultUtils.getRowsCols(before)[0], FaultUtils.getRowsCols(before)[1]), "kRainBow");
		int[] widthheight = FaultUtils.getRowsCols(a);
		int w = widthheight[1];// 112 * 4;
		int h = widthheight[0];// 6 * 100;
		int gridWidth = widthheight[1];
		int gridHeight = widthheight[0];
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

		NativeImageLoader imageLoader = new NativeImageLoader();
		// INDArray b = a.reshape(ArrayUtil.combine(new long[] { 1 },
		// a.shape()));
		// b = FaultUtils.scaleImage(b, 6, 112);

		// FaultUtils.draw(a);
		// FaultUtils.draw(sl.getSegmentationLabels());
		// FaultUtils.draw(b);
		Mat mat = imageLoader.asMat(a, 0);
		// Mat mat =
		// imageLoader.asMat(FaultUtils.zeroBorder(a));

		Mat convertedMat = new Mat();
		// mat.convertTo(convertedMat, CV_8U, 1, 0);
		mat.convertTo(convertedMat, 0, 1, 0);

		Mat image = new Mat();

		resize(convertedMat, image, new Size(w, h));

		List<Fault> aFaults = this.getFaultList();

		FaultSetter.scaleFaults(aFaults, this, preferredImageSize);

		for (Fault fault : aFaults) {
			fault.printWireInformation();
			FaultCoordinates c = fault.getFaultCoordinates();
			double[] xy1 = new double[] { c.getXCenterPixels() - (c.getXMax() - c.getXMin()) / 2.0,
					c.getYCenterPixels() - (c.getYMax() - c.getYMin()) / 2.0 };
			double[] xy2 = new double[] { c.getXCenterPixels() + (c.getXMax() - c.getXMin()) / 2.0,
					c.getYCenterPixels() + (c.getYMax() - c.getYMin()) / 2.0 };
			String label = fault.getSubFaultName().getFamilyOf();
			int x1 = (int) Math.round((double) w * (c.getXMin()) / (double) gridWidth);
			int y1 = (int) Math.round((double) h * (c.getYMin()) / (double) gridHeight);
			int x2 = (int) Math.round((double) w * ((c.getXMax())) / (double) gridWidth);
			int y2 = (int) Math.round((double) h * ((c.getYMax())) / (double) gridHeight);
			int centerX = (int) Math.round((double) h * (c.getXCenterPixels()) / (double) gridHeight);
			int centerY = (int) Math.round((double) h * (c.getYCenterPixels()) / (double) gridHeight);

			System.out.println(x1 + "  " + "  " + y1 + "  " + x2 + "  " + y2);
			rectangle(image, new Point(x1, y1), new Point(x2, y2), Scalar.YELLOW);
			rectangle(image, new Point(centerX, centerY), new Point(centerX, centerY), Scalar.GREEN);

			putText(image, label, new Point(x1, ThreadLocalRandom.current().nextInt(0, 2) == 1 ? centerY : y2),
					FONT_HERSHEY_DUPLEX, 0.5, Scalar.GREEN);
		}
		CanvasFrame frame = new CanvasFrame("Valididate");
		frame.setTitle(" Fault - Valididation");
		frame.setCanvasSize(w, h);
		frame.showImage(converter.convert(image));
		// try {
		// frame.waitKey();
		// } catch (InterruptedException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
	}

	private void testOneHotRep(CLASSuperlayer sl) {
		Set<String> labelSet = new HashSet<>();
		/**
		 * OK, we need all the faults loaded at once otherwise it doesn't make
		 * sense with the one-hot representation
		 */
		for (FaultNames d : sl.getDesiredFaults()) {
			labelSet.add(d.getSaveName());
		}
		List<String> faultLabels = new ArrayList<>(labelSet);
		Collections.sort(faultLabels);

		System.out.println(faultLabels.size());

		for (int i = 0; i < 10; i++) {

			List<Fault> objectsThisImg = sl.getFaultList();
			for (Fault io : objectsThisImg) {
				int labelIdx = faultLabels.indexOf(io.getSubFaultName().getSaveName());
				System.out.println(labelIdx + "  " + io.getSubFaultName().getSaveName() + "   iteration " + i);
			}
			System.out.println("#################################");
			sl = sl.getNewSuperLayer(1);
		}

	}

	public static void main(String[] args) {
		// FaultNames.PIN_BIG, FaultNames.PIN_SMALL, FaultNames.CONNECTOR_E,
		// FaultNames.CONNECTOR_THREE, FaultNames.CONNECTOR_TREE,
		// FaultNames.FUSE_A,
		// FaultNames.FUSE_B, FaultNames.FUSE_C, FaultNames.CHANNEL_ONE,
		// FaultNames.CHANNEL_TWO,
		// FaultNames.CHANNEL_THREE, FaultNames.DEADWIRE, FaultNames.HOTWIRE
		FaultNames desiredFault = FaultNames.DEADWIRE;
		CLASSuperlayer sl = CLASSuperlayer.builder().superlayer(1).nchannels(1).minFaults(2).maxFaults(5)
				.desiredFaults(Stream
						.of(FaultNames.CHANNEL_ONE, FaultNames.CHANNEL_TWO, FaultNames.CONNECTOR_E,
								FaultNames.CONNECTOR_TREE, FaultNames.CONNECTOR_THREE, FaultNames.FUSE_A,
								FaultNames.FUSE_B, FaultNames.FUSE_C, FaultNames.PIN_BIG, FaultNames.PIN_SMALL,
								FaultNames.HOTWIRE, FaultNames.DEADWIRE)
						.collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).isScaled(false).desiredFault(desiredFault).desiredFaultGenRate(1.0).build();
		for (int i = 0; i < 1; i++) {
			INDArray test = Nd4j.zeros(2, 113);
			INDArray test2 = Nd4j.zeros(2, 113);
			INDArray test3 = Nd4j.zeros(2, 113);

			// sl.testDrawFaults(sl);
			INDArray labels = sl.faultLocationLabels().get(desiredFault);
			System.out.println(labels);
			System.out.println("###########################################################################");
			System.out.println("###########################################################################");
			System.out.println("###########################################################################");

			INDArrayIndex[] indexs = new INDArrayIndex[] { NDArrayIndex.interval(0, 1), NDArrayIndex.interval(0, 112) };
			INDArrayIndex[] indexs2 = new INDArrayIndex[] { NDArrayIndex.interval(1, 2),
					NDArrayIndex.interval(0, 112) };

			test.put(indexs, labels.getRow(0));
			test.put(indexs2, labels.getRow(1));
			INDArrayIndex[] indexs3 = new INDArrayIndex[] { NDArrayIndex.interval(0, 1),
					NDArrayIndex.interval(112, 113) };
			INDArrayIndex[] indexs4 = new INDArrayIndex[] { NDArrayIndex.interval(1, 2),
					NDArrayIndex.interval(112, 113) };
			if ((double) labels.get(indexs).sumNumber() == 0.0) {
				test.put(indexs3, 1.0);
			}
			if ((double) labels.get(indexs2).sumNumber() == 0.0) {
				test.put(indexs4, 1.0);
			}
			// if ((double) labels.getRow(1).sumNumber() == 0.0) {
			// test.putScalar(1, 113, 1.0);
			// }
			// System.out.println(labels.get(indexs));

			System.out.println(test);
			sl = sl.getNewSuperLayer(3);
		}

		// for (int ij = 0; ij < 1; ij++) {
		// FaultSlidingInformation faultSlidingInformation =
		// desiredFault.getFSlidingInformation();
		// INDArrayIndex[] indexs;
		// INDArray image = sl.getImage().getImage().dup();
		// image = image.reshape(ArrayUtil.combine(new long[] { 1 },
		// image.shape()));
		// FaultUtils.draw(image, "Image " + ij);
		//
		// INDArray labels = sl.faultLocationLabels().get(desiredFault);
		// for (int i = 0; i < faultSlidingInformation.getXLength(); i++) {
		// indexs = new INDArrayIndex[] { NDArrayIndex.all(),
		// NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex
		// .interval(faultSlidingInformation.getXStart()[i],
		// faultSlidingInformation.getXEnd()[i]) };
		// INDArray slice = image.get(indexs).dup();
		// FaultUtils.draw(slice, "Slice " + i, "kVisibleSpectrum");
		// System.out.println(labels.getInt(0, i));
		// }
		// System.out.println(labels);
		//
		// sl = sl.getNewSuperLayer(3);
		// System.out.println("########");
		// }

	}

}
