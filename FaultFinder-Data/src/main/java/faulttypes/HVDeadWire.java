package faulttypes;

import java.util.concurrent.ThreadLocalRandom;

public class HVDeadWire extends FaultData {

	public HVDeadWire() {
		this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		makeDataSet();
	}

	@Override
	protected void makeDataSet() {
		for (int i = 0; i < data[0].length; i++) { // i are the rows (layers)
			for (int j = 0; j < data.length; j++) { // j are the columns (wires)
				if (i == (yRnd - 1) && j <= (xRnd - 1) && j >= (xRnd - 1)) {
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

	private int makeRandomData(int rangeMin, int rangeMax) {
		return ThreadLocalRandom.current().nextInt(rangeMin, rangeMax + 1);
	}
}
