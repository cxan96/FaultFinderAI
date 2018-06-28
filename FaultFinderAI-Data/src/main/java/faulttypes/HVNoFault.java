package faulttypes;

import java.util.concurrent.ThreadLocalRandom;

import utils.ArrayUtilities;

public class HVNoFault extends FaultData {

	public HVNoFault() {
		makeDataSet();
		this.label = ArrayUtilities.hvNoWireFault;
		this.reducedLabel = ArrayUtilities.hvReducedNoFault;
		this.faultLocation = -1000;

		makeDataSet();
		makeFaultArray();
	}

	@Override
	protected void makeDataSet() {
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				data[j][i] = makeRandomData(rangeMin, rangeMax);
			}
		}
	}

	private int makeRandomData(int rangeMin, int rangeMax) {
		return ThreadLocalRandom.current().nextInt(rangeMin, rangeMax + 1);
	}

	private void makeFaultArray() {
		makeReducedLabel();
	}

	/**
	 * No fault means all labels are zero. There really shouldn't be a label or
	 * reduced label for this This is only for "if need" purposes"
	 */

	private void makeReducedLabel() {
		reducedLabel[0] = 1;
	}

	public static void main(String[] args) {
		FaultData faultData = new HVNoFault();
		faultData.plotData();
		System.out.println(faultData.getXRnd());
		int[] array = faultData.getLabel();
		System.out.println(array.length);
		for (int i = 0; i < array.length; i++) {
			System.out.print(array[i] + " ");
		}
		System.out.println("\n" + faultData.getReducedLabel()[0]);
	}

}
