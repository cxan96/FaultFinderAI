package faulttypes;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;

import utils.ArrayUtilities;

public class HVChannelFault extends FaultData {

	private List<Pair<Integer, Integer>> hvChannelSegmentation = new ArrayList<Pair<Integer, Integer>>();;
	private int[] hvFaultLabel;
	private int faultLocation;

	public HVChannelFault() {
		setuphvChannelSegmentation();
		this.hvFaultLabel = ArrayUtilities.hvChannelFault;
		this.xRnd = ThreadLocalRandom.current().nextInt(0, hvChannelSegmentation.size());
		makeDataSet();
		makeFaultArray();
	}

	@Override
	public void makeDataSet() {
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				if (j <= getRandomPair().getRight() - 1 && j >= getRandomPair().getLeft() - 1) {
					data[j][i] = makeRandomData(faultRangeMin, faultRangeMax);
				} else {
					data[j][i] = makeRandomData(rangeMin, rangeMax);
				}
				// System.out.print(data[j][i] + "\t");
				// if (j == data.length - 1) {
				// System.out.println("");
				// }
			}
		}
	}

	@Override
	protected int[] getFaultLabel() {
		return hvFaultLabel;
	}

	private void makeFaultArray() {
		this.faultLocation = this.xRnd;
		for (int i = 0; i < hvFaultLabel.length; i++) {
			if (i == this.xRnd) {
				hvFaultLabel[i] = 1;
			} else {
				hvFaultLabel[i] = 0;
			}
		}
	}

	private int makeRandomData(int rangeMin, int rangeMax) {
		return ThreadLocalRandom.current().nextInt(rangeMin, rangeMax + 1);
	}

	private Pair<Integer, Integer> getRandomPair() {
		return this.hvChannelSegmentation.get(this.xRnd);
	}

	private void setuphvChannelSegmentation() {
		hvChannelSegmentation.add(Pair.of(1, 8));
		hvChannelSegmentation.add(Pair.of(9, 16));
		hvChannelSegmentation.add(Pair.of(17, 24));
		hvChannelSegmentation.add(Pair.of(25, 32));
		hvChannelSegmentation.add(Pair.of(33, 48));
		hvChannelSegmentation.add(Pair.of(49, 64));
		hvChannelSegmentation.add(Pair.of(65, 80));
		hvChannelSegmentation.add(Pair.of(81, 112));
	}

	public int getFaultLocation() {
		return this.faultLocation;
	}

	public static void main(String[] args) {
		HVChannelFault hvChannelFault = new HVChannelFault();
		int[] test = hvChannelFault.getFaultLabel();
		for (int i = 0; i < test.length; i++) {
			System.out.println(hvChannelFault.getXRand() + "    " + test[i]);
		}
	}
}
