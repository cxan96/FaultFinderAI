package test;

import java.util.concurrent.ThreadLocalRandom;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import faultTypes.FaultFactory;
import strategies.FaultRecordScalerStrategy;
import strategies.MaxStrategy;
import strategies.MinMaxStrategy;
import strategies.SigmoidStrategy;
import strategies.StandardScore;
import strategies.StandardizeMinMax;

public class TestScalerStategy {

	private int[][] data;
	private int height;
	private int width;
	private FaultFactory factory = null;
	private int faultNum;

	public TestScalerStategy() {
		this.faultNum = 1;
		init();
	}

	public TestScalerStategy(int faultNum) {
		this.faultNum = faultNum;
		init();
	}

	private void init() {
		this.factory = new FaultFactory();
		this.data = factory.getFault(faultNum).getData();
		this.height = data[0].length;
		this.width = data.length;
		// makeData();
		System.out.println(width + "  " + height);
	}

	private void makeData() {

		for (int i = 0; i < height; i++) { // i are the rows (layers)
			for (int j = 0; j < width; j++) { // j are the columns (wires)
				int fillNum;
				if ((i > height - 2) && (j > width - 3)) {
					fillNum = ThreadLocalRandom.current().nextInt(50);

				} else {
					fillNum = ThreadLocalRandom.current().nextInt(300, 500);
				}
				data[j][i] = fillNum;
				System.out.println(data[j][i] + " j = " + j + "  i = " + i);

			}
		}
	}

	public void plotData() {
		TCanvas canvas = new TCanvas("Training Data", 800, 1200);
		H2F hData = new H2F("Training Data", width, 1, width, height, 1, height);
		for (int i = 0; i < height; i++) { // i are the rows (layers)
			for (int j = 0; j < width; j++) { // j are the columns (wires)
				// System.out.println(data[j][i]);
				hData.setBinContent(j, i, data[j][i]);
			}
		}
		canvas.draw(hData);

	}

	public void plotData(FaultRecordScalerStrategy strategy) {

		INDArray features = NDArrayUtil.toNDArray(ArrayUtil.flatten(this.data));
		strategy.normalize(features);
		String canvasTitle = strategy.getClass().getName();
		double[][] data = new double[width][height];

		int rowPlacer = 0;
		int columnPlacer = 0;
		for (int i = 0; i < features.length(); i++) {
			double aDub = features.getDouble(i);
			if ((i + 1) % height == 0) {
				data[columnPlacer][rowPlacer] = aDub;
				rowPlacer = 0;
				columnPlacer++;
			} else {
				data[columnPlacer][rowPlacer] = aDub;
				rowPlacer++;

			}
		}
		TCanvas canvas = new TCanvas(canvasTitle, 800, 1200);
		H2F hData = new H2F("Training Data", width, 1, width, height, 1, height);
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				System.out.println(data[j][i]);
				hData.setBinContent(j, i, data[j][i]);
			}
		}
		canvas.draw(hData);

	}

	public static void main(String[] args) {
		TestScalerStategy test = new TestScalerStategy(3);
		test.plotData();
		System.out.println("############# MaxStrategy ###############");
		test.plotData(new MaxStrategy());

		System.out.println("############# MinMaxStrategy ###############");
		test.plotData(new MinMaxStrategy());

		System.out.println("############# StandardScore ###############");
		test.plotData(new StandardScore());

		System.out.println("############# Sigmoid ###############");
		test.plotData(new SigmoidStrategy());

		System.out.println("############# StandardizeMinMax ###############");
		test.plotData(new StandardizeMinMax(0.05));

	}
}
