package faults;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;
import java.util.zip.DataFormatException;

import org.jlab.groot.data.H2F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import utils.FaultUtils;

public class FaultFactory {
	private final int rangeMax = FaultUtils.RANGE_MAX;
	private final int rangeMin = FaultUtils.RANGE_MIN;

	private final int faultRangeMax = FaultUtils.FAULT_RANGE_MAX;
	private final int faultRangeMin = FaultUtils.FAULT_RANGE_MIN;

	private List<Fault> faultList;

	/**
	 * superLayer is used to call the correct background data, This data has
	 * been engineered from actual data in Run 3923
	 */
	private int superLayer;

	/**
	 * data is an array to generate faults into. Its ideal for each superLayer
	 */
	private int[][] data;

	/**
	 * List<Double> lMinMax is a list that contains the data minimum value and
	 * maximum value
	 */
	private List<Integer> lMinMax;
	/**
	 * randomSuperlayer is whether or not to randomize the superlayer or use the
	 * user selected superlayer idea
	 */
	private boolean randomSuperlayer;

	/**
	 * nFaults is used to generate the number of background faults to
	 * differentiate against
	 */
	private int nFaults;

	/**
	 * randomSmear is to blurr out the faults by the median value of the
	 * activations from the surrounding neighbors
	 */
	private boolean randomSmear;

	/**
	 * desiredFault is used to check if the fault to learn from was generated
	 */
	private FaultNames desiredFault;

	public FaultFactory(int superLayer, int maxFaults, FaultNames desiredFault, boolean randomSuperlayer) {
		this(superLayer, maxFaults, desiredFault, randomSuperlayer, false);
	}

	public FaultFactory(int superLayer, int maxFaults, FaultNames desiredFault, boolean randomSuperlayer,
			boolean randomSmear) {
		this.superLayer = superLayer;
		this.nFaults = ThreadLocalRandom.current().nextInt(0, maxFaults + 1);
		this.desiredFault = desiredFault;
		this.randomSuperlayer = randomSuperlayer;
		this.randomSmear = randomSmear;
		this.faultList = new ArrayList<>();
		if (!randomSuperlayer) {
			this.superLayer = ThreadLocalRandom.current().nextInt(1, 7);
		}
		loadData();
		generateFaults();
		makeDataSet();

	}

	private void loadData() {
		try {
			this.data = new int[112][6];
			int[][] newData = FaultUtils.getData(this.superLayer);
			for (int row = 0; row < 112; row++) {
				for (int col = 0; col < 6; col++) {
					this.data[row][col] = newData[row][col];
				}
			}
			this.lMinMax = getMinMax(data);
		} catch (DataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private void generateFaults() {
		for (int i = 0; i < nFaults; i++) {
			Fault fault = this.getFault();
			checkNeighborhood(fault);
		}
	}

	private Fault getFault() {
		return this.getFault(ThreadLocalRandom.current().nextInt(0, 6));
	}

	private Fault getFault(int type) {
		Fault retFault = null;

		if (type == 0) {
			retFault = new HVPinFault().getInformation();
		} else if (type == 1) {
			retFault = new HVChannelFault().getInformation();
		} else if (type == 2) {
			retFault = new HVConnectorFault().getInformation();
		} else if (type == 3) {
			retFault = new HVFuseFault().getInformation();
		} else if (type == 4) {
			retFault = new HVDeadWire().getInformation();
		} else if (type == 5) {
			retFault = new HVHotWire().getInformation();
		}
		retFault.setRandomSmear(randomSmear);
		return retFault;
	}

	private void checkNeighborhood(Fault newFault) {
		if (faultList.size() == 0) { // faultList was empty, no neighbor problem
			faultList.add(newFault);
		} else {
			boolean addFault = true;
			for (Fault fault : faultList) {// get each fault in the list already
				if (!fault.compareFault(newFault)) {
					addFault = false;
					break;
				}
			}
			if (!addFault) {

				newFault = this.getFault();
				checkNeighborhood(newFault, 0);
			}
			if (addFault) {
				faultList.add(newFault);
			}
		}
	}

	private void checkNeighborhood(Fault newFault, int runs) {
		runs++;
		boolean addFault = true;
		for (Fault fault : faultList) {// get each fault in the list already
			if (!fault.compareFault(newFault)) {
				addFault = false;
				break;
			}
		}
		if (!addFault) {
			if (runs < 30) {
				newFault = this.getFault();
				checkNeighborhood(newFault, runs);
			}
		}
		if (addFault) {
			faultList.add(newFault);
		}

	}

	private void makeDataSet() {
		for (Fault fault : this.faultList) {
			this.data = fault.placeFault(data, lMinMax);
		}
	}

	private List<Integer> getMinMax(int[][] data) {
		List<Integer> lMinMax = new ArrayList<>();
		IntStream stream = Arrays.stream(data).flatMapToInt(Arrays::stream);
		int max = stream.max().getAsInt();
		IntStream stream2 = Arrays.stream(data).flatMapToInt(Arrays::stream);
		int min = stream2.min().getAsInt();
		lMinMax.add(min);
		lMinMax.add(max);
		return lMinMax;
	}

	public void draw() {
		FaultUtils.draw(this.data);
	}

	public H2F getHist() {
		return FaultUtils.getHist(this.data);
	}

	public void printFaultList() {
		faultList.forEach(k -> {
			k.printWireInformation();
			k.getFaultCoordinates().printFaultCoordinates();
		});
	}

	public void printFaultLocation() {
		faultList.forEach(k -> {
			k.getFaultCoordinates().printFaultCoordinates();
		});
	}

	public INDArray getFeatureVector() {
		return NDArrayUtil.toNDArray(ArrayUtil.flatten(this.data));
	}

	public INDArray getFeatureVectorAsMatrix() {
		return Nd4j.create(FaultUtils.convertToDouble(this.data));
	}

	public double[][] getDataAsMatrix() {
		return FaultUtils.convertToDouble(this.data);
	}

	public int[] getFaultLabel() {
		int[] label = new int[2];
		// lets see if the desired fault is located in the list, if it is, we
		// have the label
		// [1,0]
		// If not the label is
		// [0,1]
		if (faultList.size() == 0) {
			label = IntStream.of(0, 1).toArray();
		} else {
			boolean wantedFound = false;
			for (Fault fault : faultList) {
				if (fault.getSubFaultName().equals(this.desiredFault)) {
					wantedFound = true;
				}
			}
			if (wantedFound) {
				label = IntStream.of(1, 0).toArray();
			} else {
				label = IntStream.of(0, 1).toArray();
			}
		}
		return label;

	}

	public int getSuperLayer() {
		return this.superLayer;
	}

	public List<Fault> getFaultList() {
		return this.faultList;
	}

	public int getNFaults() {
		return this.nFaults;
	}

	public static void main(String[] args) throws DataFormatException {
		TCanvas canvas = new TCanvas("aName", 800, 1200);
		canvas.divide(3, 3);
		// for (int i = 1; i < 10; i++) {
		FaultFactory factory = new FaultFactory(3, 10, FaultNames.PIN_SMALL, true, true);
		INDArray array = factory.getFeatureVectorAsMatrix();
		double[][] data = factory.getDataAsMatrix();
		FaultUtils.draw(data);
		factory.draw();
		FaultUtils.draw(array);

		int xLength = array.columns();
		int yLength = array.rows();
		int xDataLength = data[0].length;
		int yDataLength = data.length;

		System.out.println(xDataLength + "  " + xLength);
		System.out.println(yDataLength + "  " + yLength);

		for (int y = 0; y < yLength; y++) {
			for (int x = 0; x < xLength; x++) {
				System.out.println(array.getDouble(y, x) - data[y][x]);
			}
		}
		// System.out.println("columns " + array.columns() + " rows " +
		// array.rows());
		//
		// System.out.println("####################################");
		// System.out.println("##############" + factory.getSuperLayer() +
		// "#################");
		// System.out.println("####################################");
		// System.out.println("####################################");
		// factory.printFaultList();
		// System.out.println(Arrays.toString(factory.getFaultLabel()));
		canvas.cd(1);
		//
		canvas.draw(factory.getHist());
		// System.out.println(
		// factory.getFeatureVectorAsMatrix().columns() + " " +
		// factory.getFeatureVectorAsMatrix().rows());
		// System.out.println("#######%%%%%%%%%%#######%%%%%%%%%%#######%%%%%%%%%%");
		// for (int ii = 0; ii <
		// factory.getFeatureVectorAsMatrix().columns(); ii++) {
		// for (int j = 0; j < factory.getFeatureVectorAsMatrix().rows();
		// j++) {
		// System.out.println(
		// factory.getFeatureVectorAsMatrix().getDouble(j, ii) + " " +
		// FaultUtils.getData(3)[j][ii]);
		// }
		// }
		// }

	}
}// end
	// of
	// FaultFactory
	// class.
