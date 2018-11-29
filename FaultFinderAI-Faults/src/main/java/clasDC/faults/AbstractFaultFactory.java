/**
 * 
 */
package clasDC.faults;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;
import java.util.zip.DataFormatException;

import org.datavec.image.data.Image;
import org.jlab.groot.data.H2F;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import lombok.AccessLevel;
import lombok.Getter;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
@Getter
public abstract class AbstractFaultFactory {
	/**
	 * nChannels is the number of channels the return data will be in 1 ->
	 * Black&White (BW) 3-> RGB 0-255
	 */
	@Getter
	protected int nChannels;
	/**
	 * singleFaultGen used if only one fault it to be generated
	 */
	protected boolean singleFaultGen;

	/**
	 * ArrayList of Faults to be returned with the Factory
	 */
	protected List<Fault> faultList;

	/**
	 * superLayer is used to call the correct background data, This data has been
	 * engineered from actual data in Run 3923
	 */
	protected int superLayer;

	/**
	 * data is an array to generate faults into. Its ideal for each superLayer
	 */
	@Getter(AccessLevel.NONE)
	protected int[][] data;
	/**
	 * retData is an array to of faults that is returned in CLAS the coordinate <br>
	 * system of x = wires y = layers
	 */
	@Getter(AccessLevel.NONE)
	protected int[][] retData;
	/**
	 * List<Double> lMinMax is a list that contains the data minimum value and
	 * maximum value
	 */
	@Getter(AccessLevel.NONE)
	protected List<Integer> lMinMax;
	/**
	 * randomSuperlayer is whether or not to randomize the superlayer or use the
	 * user selected superlayer idea
	 */
	protected boolean randomSuperlayer;

	/**
	 * nFaults is used to generate the number of background faults to differentiate
	 * against
	 */
	protected int nFaults;

	/**
	 * maxFaults total number of faults the user would want to see in the data-set
	 * the number generated will be randomized until thi number i.e. this.nFaults =
	 * ThreadLocalRandom.current().nextInt(1, maxFaults + 1);
	 */
	protected int maxFaults;
	/**
	 * randomSmear is to blurr out the faults by the median value of the activations
	 * from the surrounding neighbors
	 */
	protected boolean randomSmear;

	/**
	 * desiredFault is used to check if the fault to learn from was generated
	 */
	protected FaultNames desiredFault;
	/**
	 * desiredFaults is a list of user defined faults. used to check if the fault to
	 * learn from was generated
	 */
	protected List<FaultNames> desiredFaults = null;

	/**
	 * labelInt place of label
	 */
	protected int labelInt;

	protected void initialize() {
		Preconditions.checkNotNull(desiredFaults,
				"The list of desired faults cannot be null. Please add faults to the list in the builder desiredFault(List<FaultNames>)");
		this.nFaults = ThreadLocalRandom.current().nextInt(0, maxFaults + 1);
		if (randomSuperlayer) {
			this.superLayer = ThreadLocalRandom.current().nextInt(1, 7);
		}
		this.faultList = new ArrayList<>();
	}

	protected void loadData() {
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

	// private Fault getFault() {
	// if (singleFaultGen) {
	// return this.getFault(desiredFault);
	// }
	// return this.getFault(ThreadLocalRandom.current().nextInt(0, 4));
	// }

	protected abstract Fault getFault();

	protected Fault getFault(int type) {
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

	protected Fault getFault(FaultNames type) {
		Fault retFault = null;

		if (type.equals(FaultNames.PIN_SMALL) || type.equals(FaultNames.PIN_BIG)) {
			retFault = new HVPinFault(type).getInformation();
		} else if (type.equals(FaultNames.CHANNEL_ONE) || type.equals(FaultNames.CHANNEL_TWO)
				|| type.equals(FaultNames.CHANNEL_THREE)) {
			retFault = new HVChannelFault(type).getInformation();
		} else if (type.equals(FaultNames.CONNECTOR_E) || type.equals(FaultNames.CONNECTOR_THREE)
				|| type.equals(FaultNames.CONNECTOR_TREE)) {
			retFault = new HVConnectorFault(type).getInformation();
		} else if (type.equals(FaultNames.FUSE_A) || type.equals(FaultNames.FUSE_B) || type.equals(FaultNames.FUSE_C)) {
			retFault = new HVFuseFault(type).getInformation();
		} else if (type.equals(FaultNames.DEADWIRE)) {
			retFault = new HVDeadWire().getInformation();
		} else if (type.equals(FaultNames.HOTWIRE)) {
			retFault = new HVHotWire().getInformation();
		}
		retFault.setRandomSmear(randomSmear);
		return retFault;
	}

	protected void generateFaults() {
		for (int i = 0; i < nFaults; i++) {
			Fault fault = this.getFault();
			checkNeighborhood(fault);
		}
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

	protected void makeDataSet() {
		for (Fault fault : this.faultList) {
			this.data = fault.placeFault(data, lMinMax);
		}
	}

	protected void convertDataset() {
		this.retData = new int[6][112];
		for (int i = 0; i < data[0].length; i++) {
			for (int j = 0; j < data.length; j++) {
				// hData.setBinContent(j, i, data[j][i]);
				this.retData[i][j] = this.data[j][i];
			}
		}

	}

	protected abstract int[] getFaultLabel();

	// public everyday stuff
	public abstract AbstractFaultFactory getNewFactory();

	public abstract AbstractFaultFactory getNewFactory(int superLayer);

	public void draw() {
		FaultUtils.draw(this.retData);
	}

	public H2F getHist() {
		return FaultUtils.getHist(this.retData);
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
		return NDArrayUtil.toNDArray(ArrayUtil.flatten(this.retData));
	}

	public INDArray getFeatureVectorAsMatrix() {

		return Nd4j.create(FaultUtils.convertToDouble(this.retData));
	}

	public Image asImageMatrix() {
		return FaultUtils.asImageMatrix(this.nChannels, getFeatureVectorAsMatrix());
	}

	public Image asImageMatrix(int dimensions) {
		return FaultUtils.asImageMatrix(dimensions, this.nChannels, getFeatureVectorAsMatrix());
	}

	public Image asUnShapedImageMatrix() {

		return FaultUtils.asUnShapedImageMatrix(this.nChannels, getFeatureVectorAsMatrix());
	}

	public double[][] getDataAsMatrix() {
		return FaultUtils.convertToDouble(this.retData);
	}

}
