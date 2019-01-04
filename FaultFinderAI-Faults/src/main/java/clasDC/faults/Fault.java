package clasDC.faults;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import clasDC.factories.CLASComponent;
import clasDC.factories.CLASDCRegion;
import clasDC.factories.CLASDCSystem;
import clasDC.factories.CLASDriftChamber;
import clasDC.factories.CLASSuperlayer;
import lombok.Getter;
import utils.FaultUtils;

public class Fault {
	/**
	 * Map<layerLocation, Pair<leftWire,rightWire>>
	 */
	private Map<Integer, Pair<Integer, Integer>> wireInfo;
	private String faultName;
	private FaultNames subFaultName;
	private boolean randomSmear;
	private Set<FaultNames> notSixLayerFaults;
	private FaultCoordinates faultCoordinates;
	@Getter
	private Pair<Integer, Integer> placement;

	public Fault(String faultName, FaultNames subFaultName, Map<Integer, Pair<Integer, Integer>> wireInfo) {
		this.faultName = faultName;
		this.subFaultName = subFaultName;
		this.wireInfo = wireInfo;
		this.randomSmear = false;
		makeSixLayerFaults();
		setFaultCoordinates();
	}

	public Fault(String faultName, FaultNames subFaultName, Map<Integer, Pair<Integer, Integer>> wireInfo,
			Pair<Integer, Integer> placement) {
		this.faultName = faultName;
		this.subFaultName = subFaultName;
		this.wireInfo = wireInfo;
		this.randomSmear = false;
		this.placement = placement;
		makeSixLayerFaults();
		setFaultCoordinates();
	}

	private void makeSixLayerFaults() {
		this.notSixLayerFaults = new HashSet<>();
		notSixLayerFaults.add(FaultNames.HOTWIRE);
		notSixLayerFaults.add(FaultNames.DEADWIRE);
		notSixLayerFaults.add(FaultNames.PIN_BIG);
		notSixLayerFaults.add(FaultNames.PIN_SMALL);
	}

	public Map<Integer, Pair<Integer, Integer>> getWireInfo() {
		return this.wireInfo;
	}

	public String getFaultName() {
		return this.faultName;
	}

	public FaultNames getSubFaultName() {
		return this.subFaultName;
	}

	public boolean compareFault(Fault anotherFault) {

		for (Map.Entry<Integer, Pair<Integer, Integer>> entry : this.getWireInfo().entrySet()) {
			Integer key = entry.getKey();
			Pair<Integer, Integer> value = entry.getValue();
			for (Map.Entry<Integer, Pair<Integer, Integer>> anotherEntry : anotherFault.getWireInfo().entrySet()) {
				Integer anotherKey = anotherEntry.getKey();
				Pair<Integer, Integer> anotherValue = anotherEntry.getValue();
				if (anotherKey.equals(key)) {
					// now check to see if the new fault is inside the existing
					// fault
					if (anotherValue.getLeft() >= value.getLeft() && anotherValue.getLeft() <= value.getRight()) {
						return false;
					} // this works
						// lets make 2 statements so its more readable
					if (anotherValue.getRight() >= value.getLeft() && anotherValue.getRight() <= value.getRight()) {
						return false;
					}
					if (anotherValue.getRight() >= value.getRight() && anotherValue.getLeft() <= value.getLeft()) {
						return false;
					}
					if (anotherValue.getRight() == value.getRight() && anotherValue.getLeft() == value.getLeft()) {
						return false;
					}
				}
			}
		}
		return true;
	}

	public void printWireInformation() {
		System.out.println(this.getFaultName() + "  " + this.getSubFaultName());

		this.getWireInfo().forEach((k, v) -> {
			System.out.println("thisFault Layer " + k + " with left: " + v.getLeft() + " with right: " + v.getRight());
		});
		this.faultCoordinates.printFaultCoordinates();
	}

	public int[][] placeFault(int[][] data, List<Integer> lMinMax) {

		int min;
		int max;
		if (randomSmear) {
			int smearValue;
			// Deadwire has to be different since its not a collection of
			// activations
			if (this.subFaultName.equals(FaultNames.DEADWIRE)) {
				smearValue = ThreadLocalRandom.current().nextInt(5, 7);
			} else if (this.subFaultName.equals(FaultNames.HOTWIRE)) {
				// smearValue = ThreadLocalRandom.current().nextInt(100, 200);
				smearValue = ThreadLocalRandom.current().nextInt(200, 300);

			} else {
				smearValue = ThreadLocalRandom.current().nextInt(5, 85);
			}
			// smearValue = ThreadLocalRandom.current().nextInt(5, 15);

			double lowValue;
			double highValue;
			if (smearValue == 0.0) {
				lowValue = 0.0;
				highValue = 0.05;
			} else {
				lowValue = ((double) smearValue) / 100.0 - 0.05;
				highValue = ((double) smearValue) / 100.0;
			}
			int averageNeighbors = averageNeighbors(data);
			min = (int) (lowValue * averageNeighbors);
			max = (int) (highValue * averageNeighbors);
			// System.out.println(
			// min + " MIN and MAX " + max + " " + this.faultName + "
			// averageNeighbors " + averageNeighbors);
			// System.out.println(lowValue + " lowValue and highValue " +
			// highValue + " " + this.faultName);

		} else {
			min = 0;
			max = 0;
			throw new IllegalArgumentException("I made it here for some reason");
		}
		// else {
		// if (this.subFaultName.equals(FaultNames.HOTWIRE)) {
		// min = lMinMax.get(1) * 2;
		// max = 4 * min;
		// } else {
		// min = 0;
		// max = lMinMax.get(0);
		// }
		// }

		for (Map.Entry<Integer, Pair<Integer, Integer>> entry : this.getWireInfo().entrySet()) {
			Integer layer = entry.getKey() - 1;
			Pair<Integer, Integer> wires = entry.getValue();
			for (int j = 0; j < data.length; j++) { // j are the columns
													// (wires)
				if (j <= wires.getRight() - 1 && j >= wires.getLeft() - 1) {
					data[j][layer] = makeRandomData(min, max);
				}
			}

		}
		return data;
	}

	public void placeFault(INDArray features, INDArray labels) {
		double min;
		double max;
		if (randomSmear) {
			int smearValue;
			// Deadwire has to be different since its not a collection of
			// activations
			if (this.subFaultName.equals(FaultNames.DEADWIRE)) {
				smearValue = ThreadLocalRandom.current().nextInt(5, 7);
			} else if (this.subFaultName.equals(FaultNames.HOTWIRE)) {
				// smearValue = ThreadLocalRandom.current().nextInt(150, 200);
				smearValue = ThreadLocalRandom.current().nextInt(200, 300);

				// smearValue = ThreadLocalRandom.current().nextInt(1400, 1600);

			} else {
				// smearValue = ThreadLocalRandom.current().nextInt(5, 85);
				smearValue = ThreadLocalRandom.current().nextInt(5, 6);
			}
			double lowValue;
			double highValue;
			if (smearValue == 0.0) {
				lowValue = 0.0;
				highValue = 0.05;
			} else {
				lowValue = ((double) smearValue) / 100.0 - 0.05;
				highValue = ((double) smearValue) / 100.0;
			}

			double averageNeighbors = averageNeighbors(features);
			// System.out.println(averageNeighbors + " averageNeighbors IND");
			min = (lowValue * averageNeighbors);
			max = (highValue * averageNeighbors);
		} else {
			min = 0;
			max = 0;
			throw new IllegalArgumentException("I made it here for some reason");
		}
		for (Map.Entry<Integer, Pair<Integer, Integer>> entry : this.getWireInfo().entrySet()) {
			Integer layer = entry.getKey() - 1;
			Pair<Integer, Integer> wires = entry.getValue();
			for (int j = 0; j < features.rows(); j++) { // j are the columns
				// (wires)
				if (j <= wires.getRight() - 1 && j >= wires.getLeft() - 1) {
					features.putScalar(j, layer, makeRandomData(min, max));
					labels.putScalar(j, layer, this.subFaultName.getLabel());

				}
			}

		}
	}

	public void setRandomSmear(boolean randomSmear) {
		this.randomSmear = randomSmear;
	}

	private int makeRandomData(double rangeMin, double rangeMax) {
		return ThreadLocalRandom.current().nextInt((int) rangeMin, (int) (rangeMax + 1));
	}

	private int averageNeighbors(int[][] data) {
		List<Integer> aList = new ArrayList<>();
		double retVal = 0.0;

		for (Map.Entry<Integer, Pair<Integer, Integer>> entry : this.getWireInfo().entrySet()) {
			Integer layer = entry.getKey() - 1;
			Pair<Integer, Integer> wires = entry.getValue();
			// add the left and right activations next to the fault
			if (wires.getLeft() != 1) {// this is the left most wire, nothing
				aList.add(data[wires.getLeft() - 2][layer]);
			}
			// this is the right most wire, nothing before this
			if (wires.getRight() != 112) {
				aList.add(data[wires.getRight()][layer]);
			}

			/**
			 * Sum below and above the fault IF the fault is not a 1-6 layer
			 * fault. Far less faults that do not span six layers
			 */
			if (this.notSixLayerFaults.contains(this.subFaultName)) {
				/**
				 * sum activations below the fault superlayer of the fault is
				 * not SL1 i.e. entry.getKey!=1
				 */
				if ((layer + 1) != 1) {
					for (int j = 0; j < data.length; j++) { // j are the columns
						// (wires)
						if (j <= wires.getRight() - 1 && j >= wires.getLeft() - 1) {
							aList.add(data[j][layer - 1]);
						}
					}
				}

				/**
				 * sum activations above the fault superlayer of the fault is
				 * not SL6 i.e. entry.getKey!=6
				 */
				if ((layer + 1) != 6) {
					for (int j = 0; j < data.length; j++) { // j are the columns
						// (wires)
						if (j <= wires.getRight() - 1 && j >= wires.getLeft() - 1) {
							aList.add(data[j][layer + 1]);
						}
					}
				}
			}
			INDArray array = Nd4j.create(aList);
			retVal = (double) array.medianNumber();
		}

		return (int) retVal;
	}

	private double averageNeighbors(INDArray data) {
		List<Integer> aList = new ArrayList<>();
		double retVal = 0.0;

		for (Map.Entry<Integer, Pair<Integer, Integer>> entry : this.getWireInfo().entrySet()) {
			Integer layer = entry.getKey() - 1;
			Pair<Integer, Integer> wires = entry.getValue();
			// add the left and right activations next to the fault
			if (wires.getLeft() != 1) {// this is the left most wire, nothing
				// aList.add(data[wires.getLeft() - 2][layer]);
				aList.add(data.getInt((wires.getLeft() - 2), layer));

			}
			// this is the right most wire, nothing before this
			if (wires.getRight() != 112) {
				// aList.add(data[wires.getRight()][layer]);
				aList.add(data.getInt(wires.getRight(), layer));

			}

			/**
			 * Sum below and above the fault IF the fault is not a 1-6 layer
			 * fault. Far less faults that do not span six layers
			 */
			if (this.notSixLayerFaults.contains(this.subFaultName)) {
				/**
				 * sum activations below the fault layer of the fault is not
				 * layer1 i.e. entry.getKey!=1
				 */
				if ((layer + 1) != 1) {
					for (int j = 0; j < data.columns(); j++) { // j are the
																// columns
						// (wires)
						if (j <= wires.getRight() - 1 && j >= wires.getLeft() - 1) {
							// aList.add(data[j][layer - 1]);
							aList.add(data.getInt(j, (layer - 1)));

						}
					}
				}

				/**
				 * sum activations above the fault slayer of the fault is not
				 * Layer6 i.e. entry.getKey!=6
				 */
				if ((layer + 1) != 6) {
					for (int j = 0; j < data.columns(); j++) { // j are the
																// columns
						// (wires)
						if (j <= wires.getRight() - 1 && j >= wires.getLeft() - 1) {
							// aList.add(data[j][layer + 1]);
							aList.add(data.getInt(j, (layer + 1)));

						}
					}
				}
			}
			INDArray array = Nd4j.create(aList);
			retVal = (double) array.medianNumber();
		}

		return retVal;
	}

	public FaultCoordinates getFaultCoordinates() {

		return this.faultCoordinates;
	}

	private void setFaultCoordinates() {
		// first let get the wireinfomap

		int xMin = 113; // 1 more than possible allowed number of wires
		int xMax = 0;
		int yMin = 7; // 1 more than possible allowed number of layers
		int yMax = 0;
		for (Map.Entry<Integer, Pair<Integer, Integer>> entry : this.getWireInfo().entrySet()) {
			Integer key = entry.getKey();
			Pair<Integer, Integer> value = entry.getValue();
			xMin = Math.min(xMin, value.getLeft());
			xMax = Math.max(xMax, value.getRight());
			yMin = Math.min(yMin, key);
			yMax = Math.max(yMax, key);
		}
		this.faultCoordinates = new FaultCoordinates(xMin - 1, yMin - 1, xMax, yMax, this.subFaultName.getSaveName());
	}

	public void offsetFaultCoodinates(double offset, String axis) {
		double xMin = this.faultCoordinates.getXMin();
		double xMax = this.faultCoordinates.getXMax();
		double yMin = this.faultCoordinates.getYMin();
		double yMax = this.faultCoordinates.getYMax();
		if (axis.toLowerCase().equals("x")) {
			xMax = xMax + offset;
			xMin = xMin + offset;

		} else if (axis.toLowerCase().equals("y")) {
			yMax = yMax + offset;
			yMin = yMin + offset;

		} else {
			throw new IllegalArgumentException("Invalid input: (x or y axis can only be changed");
		}
		this.faultCoordinates = new FaultCoordinates(xMin, yMin, xMax, yMax, this.subFaultName.getSaveName());

	}

	/**
	 * 
	 * @param int[]
	 *            scales (xScale, yScale) -> (width, height)
	 */
	public void scaleFaultCoodinates(CLASComponent comp) {
		int[] scales;
		if (comp instanceof CLASSuperlayer) {
			scales = FaultUtils.scaleImage(6, 112);
		} else if (comp instanceof CLASDriftChamber) {
			scales = FaultUtils.scaleImage(12, 112);
		} else if (comp instanceof CLASDCRegion) {
			scales = FaultUtils.scaleImage(72, 112);
		} else if (comp instanceof CLASDCSystem) {
			scales = FaultUtils.scaleImage(72 * 3, 112);
		} else {
			throw new IllegalArgumentException("Invalid input: " + comp + "  for a clasObject");
		}
		double xMin = this.faultCoordinates.getXMin() * scales[1] + scales[3];
		double xMax = this.faultCoordinates.getXMax() * scales[1] + scales[3];
		double yMin = this.faultCoordinates.getYMin() * scales[0] + scales[2];
		double yMax = this.faultCoordinates.getYMax() * scales[0] + scales[2];
		this.faultCoordinates = new FaultCoordinates(xMin, yMin, xMax, yMax, this.subFaultName.getSaveName());

	}

	/**
	 * 
	 * @param int[]
	 *            scales (xScale, yScale) -> (width, height)
	 */
	public void scaleFaultCoodinates(CLASComponent comp, int preferredImageSize) {
		int[] scales;
		if (comp instanceof CLASSuperlayer) {
			scales = FaultUtils.scaleImage(preferredImageSize, 6, 112);
		} else if (comp instanceof CLASDriftChamber) {
			scales = FaultUtils.scaleImage(preferredImageSize, 12, 112);
		} else if (comp instanceof CLASDCRegion) {
			scales = FaultUtils.scaleImage(preferredImageSize, 72, 112);
		} else if (comp instanceof CLASDCSystem) {
			scales = FaultUtils.scaleImage(preferredImageSize, 72 * 3, 112);
		} else {
			throw new IllegalArgumentException("Invalid input: " + comp + "  for a clasObject");
		}
		double xMin = this.faultCoordinates.getXMin() * scales[1] + scales[3];
		double xMax = this.faultCoordinates.getXMax() * scales[1] + scales[3];
		double yMin = this.faultCoordinates.getYMin() * scales[0] + scales[2];
		double yMax = this.faultCoordinates.getYMax() * scales[0] + scales[2];
		this.faultCoordinates = new FaultCoordinates(xMin, yMin, xMax, yMax, this.subFaultName.getSaveName());

	}

	/**
	 * 
	 * @param int[]
	 *            scales (xScale, yScale) -> (width, height)
	 */
	public void scaleFaultCoodinates(CLASComponent comp, int[] preferredImageSize) {
		int[] scales;
		if (comp instanceof CLASSuperlayer) {
			scales = FaultUtils.scaleImage(preferredImageSize, 6, 112);
		} else if (comp instanceof CLASDriftChamber) {
			scales = FaultUtils.scaleImage(preferredImageSize, 12, 112);
		} else if (comp instanceof CLASDCRegion) {
			scales = FaultUtils.scaleImage(preferredImageSize, 72, 112);
		} else if (comp instanceof CLASDCSystem) {
			scales = FaultUtils.scaleImage(preferredImageSize, 72 * 3, 112);
		} else {
			throw new IllegalArgumentException("Invalid input: " + comp + "  for a clasObject");
		}
		double xMin = this.faultCoordinates.getXMin() * scales[1] + scales[3];
		double xMax = this.faultCoordinates.getXMax() * scales[1] + scales[3];
		double yMin = this.faultCoordinates.getYMin() * scales[0] + scales[2];
		double yMax = this.faultCoordinates.getYMax() * scales[0] + scales[2];
		this.faultCoordinates = new FaultCoordinates(xMin, yMin, xMax, yMax, this.subFaultName.getSaveName());

	}

	/**
	 * 
	 * @param int[]
	 *            scales (xScale, yScale) -> (width, height)
	 * @param String
	 *            String:
	 *            "CLASSuperlayer","CLASDriftChamber","CLASDCRegion","CLASDCSystem"
	 *
	 */
	public void scaleFaultCoodinates(String comp, int[] preferredImageSize) {
		int[] scales;
		if (comp.equals("CLASSuperlayer")) {
			scales = FaultUtils.scaleImage(preferredImageSize, 6, 112);
		} else if (comp.equals("CLASDriftChamber")) {
			scales = FaultUtils.scaleImage(preferredImageSize, 12, 112);
		} else if (comp.equals("CLASDCRegion")) {
			scales = FaultUtils.scaleImage(preferredImageSize, 72, 112);
		} else if (comp.equals("CLASDCSystem")) {
			scales = FaultUtils.scaleImage(preferredImageSize, 72 * 3, 112);
		} else {
			throw new IllegalArgumentException("Invalid input: " + comp + "  for a clasObject");
		}
		double xMin = this.faultCoordinates.getXMin() * scales[1] + scales[3];
		double xMax = this.faultCoordinates.getXMax() * scales[1] + scales[3];
		double yMin = this.faultCoordinates.getYMin() * scales[0] + scales[2];
		double yMax = this.faultCoordinates.getYMax() * scales[0] + scales[2];
		this.faultCoordinates = new FaultCoordinates(xMin, yMin, xMax, yMax, this.subFaultName.getSaveName());

	}
}
