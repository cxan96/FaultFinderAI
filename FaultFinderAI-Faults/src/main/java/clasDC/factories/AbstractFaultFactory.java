/**
 * 
 */
package clasDC.factories;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.zip.DataFormatException;

import org.datavec.image.data.Image;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import clasDC.faults.Fault;
import clasDC.faults.FaultCoordinates;
import clasDC.faults.FaultNames;
import clasDC.faults.HVChannelFault;
import clasDC.faults.HVConnectorFault;
import clasDC.faults.HVDeadWire;
import clasDC.faults.HVFuseFault;
import clasDC.faults.HVHotWire;
import clasDC.faults.HVPinFault;
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
	 * superLayer is used to call the correct background data, This data has
	 * been engineered from actual data in Run 3923
	 */
	protected int superLayer;

	/**
	 * features is an array to of faults that is returned in CLAS the coordinate
	 * <br>
	 * system of x = wires y = layers
	 */
	private INDArray features;

	/**
	 * labels is an array to of fault labels as a figure that is returned in
	 * CLAS the coordinate <br>
	 * system of x = wires y = layers
	 */
	private INDArray labels;

	/**
	 * List<Double> lMinMax is a list that contains the data minimum value and
	 * maximum value
	 */
	@Getter(AccessLevel.NONE)
	protected List<Double> lMinMax;
	/**
	 * randomSuperlayer is whether or not to randomize the superlayer or use the
	 * user selected superlayer idea
	 */
	protected boolean randomSuperlayer;

	/**
	 * nFaults is used to generate the number of background faults to
	 * differentiate against
	 */
	protected int nFaults;

	/**
	 * maxFaults total number of faults the user would want to see in the
	 * data-set the number generated will be randomized until thi number i.e.
	 * this.nFaults = ThreadLocalRandom.current().nextInt(1, maxFaults + 1);
	 */
	protected int maxFaults;

	/**
	 * minFaults minimum number of faults the user would want to see in the
	 * data-set.Default value is 0.
	 */
	protected int minFaults = 0;

	/**
	 * randomSmear is to blurr out the faults by the median value of the
	 * activations from the surrounding neighbors
	 */
	protected boolean randomSmear;

	/**
	 * desiredFault is used to check if the fault to learn from was generated
	 */
	protected FaultNames desiredFault = null;
	/**
	 * desiredFaults is a list of user defined faults. used to check if the
	 * fault to learn from was generated
	 */
	protected List<FaultNames> desiredFaults = null;

	/**
	 * double desiredFaultGenRate: the rate at which the desiredFault is
	 * generated <br>
	 * default is 1.0 sampled from flat distribution if((double )randomNum <
	 * desiredFaultGenRate) GenDesiredFault else GenRandom fault caveat: the
	 * randomFault might be the desired fault which will skew this rate of
	 * generation
	 */
	protected double desiredFaultGenRate;

	/**
	 * labelInt place of label
	 */
	protected int labelInt;

	protected void initialize() {
		if (desiredFaults == null && desiredFault == null) {
			throw new NullPointerException(
					"The list of desired faults cannot be null. Please add faults to the list in the builder desiredFault(List<FaultNames>) <br> OR add a desiredFault");
		}

		Preconditions.checkArgument(minFaults <= maxFaults,
				"minFaults cannot be larger than maxFaults. Please check te builder: (minFaults, maxFaults)  ("
						+ minFaults + ", " + maxFaults + ")");

		// Check to see if desiredFaultGenRate == 0.0, user should never want
		// this. Defeats purpose of this setting. Therefore, the setting must
		// not have been initialized so the default desiredFaultGenRate =1.0
		// should be invoked
		if (this.desiredFaultGenRate == 0.0) {
			this.desiredFaultGenRate = 1.0;
		}

		Preconditions.checkArgument(this.desiredFaultGenRate <= 1.0,
				"This desiredFaultGenRate must be bound (0.0,1.0), you entered a desiredFaultGenRate of "
						+ this.desiredFaultGenRate);
		Preconditions.checkArgument(this.desiredFaultGenRate > 0.0,
				"This desiredFaultGenRate must be positive and bound (0.0,1.0), you entered a desiredFaultGenRate of "
						+ this.desiredFaultGenRate);

		this.nFaults = ThreadLocalRandom.current().nextInt(minFaults, maxFaults + 1);
		if (randomSuperlayer) {
			this.superLayer = ThreadLocalRandom.current().nextInt(1, 7);
		}
		this.faultList = new ArrayList<>();
	}

	protected void loadData() {
		this.labels = Nd4j.zeros(112, 6);

		try {
			this.features = Nd4j.create(FaultUtils.convertToDouble(FaultUtils.getData(this.superLayer)));

			this.lMinMax = getMinMax(this.features);
		} catch (DataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private List<Double> getMinMax(INDArray data) {
		List<Double> lMinMax = new ArrayList<>();
		lMinMax.add((Double) data.minNumber());
		lMinMax.add((Double) data.maxNumber());
		return lMinMax;
	}

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
		checkDesiredFaults();
		// check to see if user has a desired fault to simulate.
		// if yes, put this first in the list, else simulate normally
		if (this.desiredFault != null) {
			// Now we have established that the desiredFault is defined,
			// generate this according to the desiredFaultGenRate
			Fault fault;
			// System.out.println(this.desiredFaultGenRate + "
			// desiredFaultGenRate");
			if (ThreadLocalRandom.current().nextDouble(0.0, 1.0) > (1.0 - this.desiredFaultGenRate)) {
				fault = this.getFault(this.desiredFault);
			} else {
				fault = this.getFault();
			}

			checkNeighborhood(fault);
		}
		for (int i = 0; i < (this.desiredFault == null ? nFaults : (nFaults - 1)); i++) {
			Fault fault = this.getFault();
			checkNeighborhood(fault);
		}
	}

	private void checkDesiredFaults() {
		if (this.desiredFault != null) {
			if (this.desiredFaults == null) {
				this.desiredFaults = new ArrayList<>();
			}
			if (!this.desiredFaults.contains(this.desiredFault)) {
				this.desiredFaults.add(desiredFault);
			}
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
			fault.placeFault(features, labels);
		}
	}

	protected void convertDataset() {
		this.features = this.features.transpose();
		this.labels = this.labels.transpose();
	}

	protected abstract int[] getFaultLabel();

	// public everyday stuff
	public abstract AbstractFaultFactory getNewFactory();

	public abstract AbstractFaultFactory getNewFactory(int superLayer);

	public void drawFeatures() {
		FaultUtils.draw(this.features);
	}

	public void drawLabels() {
		FaultUtils.draw(this.labels);
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

	public INDArray getFeatures() {
		return this.features;
	}

	public Image asImageMatrix() {
		return FaultUtils.asImageMatrix(this.nChannels, getFeatures());
	}

	public Image asLabelImageMatrix() {
		return FaultUtils.asImageMatrix(this.nChannels, getLabels());
	}

	public Image asImageMatrix(int nChannels) {
		return FaultUtils.asImageMatrix(nChannels, getFeatures());
	}

	public Image asUnShapedImageMatrix() {

		return FaultUtils.asUnShapedImageMatrix(this.nChannels, getFeatures());
	}

	public Map<String, INDArray> locationLabels() {
		Map<String, INDArray> ret = new HashMap<>();

		for (Map.Entry<FaultNames, INDArray> entry : faultLocationLabels().entrySet()) {
			FaultNames key = entry.getKey();
			INDArray value = entry.getValue();
			ret.put(key.getSaveName(), value);
		}
		return ret;

	}

	public Map<FaultNames, INDArray> faultLocationLabels() {
		Map<FaultNames, INDArray> ret = new HashMap<>();
		ret.clear();
		for (FaultNames faultNames : desiredFaults) {
			ret.put(faultNames, faultNames.getPossiblePositions().dup());
		}

		for (Fault fault : faultList) {
			if (this.desiredFaults.contains(fault.getSubFaultName())) {
				FaultCoordinates coordinates = fault.getFaultCoordinates();
				int xCoord;
				int yCoord;
				int xTemp = (int) coordinates.getXMin();

				if (fault.getSubFaultName().equals(FaultNames.PIN_BIG)
						|| fault.getSubFaultName().equals(FaultNames.PIN_SMALL)
						|| fault.getSubFaultName().equals(FaultNames.HOTWIRE)
						|| fault.getSubFaultName().equals(FaultNames.DEADWIRE)) {
					yCoord = (int) coordinates.getYMin();
				} else {
					yCoord = 0;
				}
				if (fault.getSubFaultName().equals(FaultNames.CHANNEL_ONE)
						|| fault.getSubFaultName().equals(FaultNames.CONNECTOR_E)
						|| fault.getSubFaultName().equals(FaultNames.CONNECTOR_TREE)
						|| fault.getSubFaultName().equals(FaultNames.CONNECTOR_THREE)
						|| fault.getSubFaultName().equals(FaultNames.FUSE_A)
						|| fault.getSubFaultName().equals(FaultNames.FUSE_B)
						|| fault.getSubFaultName().equals(FaultNames.FUSE_C)
						|| fault.getSubFaultName().equals(FaultNames.PIN_SMALL)) {
					int xPlace = xTemp / 8 * 8;
					if (fault.getSubFaultName().equals(FaultNames.PIN_SMALL)
							|| fault.getSubFaultName().equals(FaultNames.CHANNEL_ONE)
							|| fault.getSubFaultName().equals(FaultNames.CONNECTOR_E)
							|| fault.getSubFaultName().equals(FaultNames.CONNECTOR_TREE)
							|| fault.getSubFaultName().equals(FaultNames.CONNECTOR_THREE)) {
						xCoord = (xPlace / 8);
					} else {
						xCoord = (xPlace / 16); // was 16
					}

				} else if (fault.getSubFaultName().equals(FaultNames.PIN_BIG)) {
					int xPlace = (xTemp - 80) / 8 * 8;
					xCoord = (xPlace / 16);
				} else if (fault.getSubFaultName().equals(FaultNames.CHANNEL_TWO)) {
					int xPlace = (xTemp - 32) / 8 * 8;
					xCoord = (xPlace / 16);
				} else if (fault.getSubFaultName().equals(FaultNames.CHANNEL_THREE)) {
					xCoord = 1;
				} else if (fault.getSubFaultName().equals(FaultNames.DEADWIRE)
						|| fault.getSubFaultName().equals(FaultNames.HOTWIRE)) {
					xCoord = xTemp;
				} else {
					throw new IllegalAccessError("This fault is not recognized");
				}
				// System.out.println(fault.getSubFaultName().getSaveName() + "
				// " + yCoord + " " + xCoord + " " + xTemp
				// + "\n ################# \n");

				ret.get(fault.getSubFaultName()).putScalar(yCoord, xCoord, 1.0);
			}
		}
		return ret;

	}

	// public Image getScaledImage() {
	// }

	public INDArray getLabels() {

		return this.labels;
	}
}
