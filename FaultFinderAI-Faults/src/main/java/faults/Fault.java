package faults;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Fault {
	/**
	 * Map<layerLocation, Pair<leftWire,rightWire>>
	 */
	private Map<Integer, Pair<Integer, Integer>> wireInfo;
	private String faultName;
	private FaultNames subFaultName;
	private boolean randomSmear;

	public Fault(String faultName, FaultNames subFaultName, Map<Integer, Pair<Integer, Integer>> wireInfo) {
		this.faultName = faultName;
		this.subFaultName = subFaultName;
		this.wireInfo = wireInfo;
		this.randomSmear = false;
	}

	public Map<Integer, Pair<Integer, Integer>> getWireInfo() {
		return wireInfo;
	}

	public String getFaultName() {
		return faultName;
	}

	public FaultNames getSubFaultName() {
		return subFaultName;
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
	}

	public int[][] placeFault(int[][] data, List<Integer> lMinMax) {

		int min;
		int max;
		if (this.subFaultName.equals(FaultNames.HOTWIRE)) {
			min = lMinMax.get(1) * 2;
			max = 10 * min;
		} else {
			if (randomSmear) {
				int smearValue;
				// Deadwire has to be different since its not a collection of
				// activations
				if (this.subFaultName.equals(FaultNames.DEADWIRE)) {
					smearValue = ThreadLocalRandom.current().nextInt(5, 40);
				} else {
					smearValue = ThreadLocalRandom.current().nextInt(5, 100);
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
				min = (int) (lowValue * averageNeighbors(data));
				max = (int) (highValue * averageNeighbors(data));
			} else {
				min = 0;
				max = lMinMax.get(0);
			}
		}

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
										// before this
				aList.add(data[wires.getLeft() - 2][layer]);
			}
			if (wires.getRight() != 112) {// this is the right most wire,
											// nothing
				// before this
				aList.add(data[wires.getRight() - 2][layer]);
			}
			// sum activations below the fault
			// superlayer of the fault is not SL1 i.e. entry.getKey!=1
			if ((layer + 1) != 1) {
				for (int j = 0; j < data.length; j++) { // j are the columns
					// (wires)
					if (j <= wires.getRight() - 1 && j >= wires.getLeft() - 1) {
						aList.add(data[j][layer - 1]);
					}
				}
			}
			// sum activations above the fault
			// superlayer of the fault is not SL6 i.e. entry.getKey!=6
			if ((layer + 1) != 6) {
				for (int j = 0; j < data.length; j++) { // j are the columns
					// (wires)
					if (j <= wires.getRight() - 1 && j >= wires.getLeft() - 1) {
						aList.add(data[j][layer + 1]);
					}
				}
			}
			INDArray array = Nd4j.create(aList);
			retVal = (double) array.medianNumber();
		}

		return (int) retVal;
	}
}
