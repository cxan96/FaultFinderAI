package faultTypes;

import java.util.HashMap;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;

public class HVHotWire extends FaultData {

	public HVHotWire() {
		this.xRnd = ThreadLocalRandom.current().nextInt(1, 113);
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		this.faultName = FaultNames.HOTWIRE;

	}

	@Override
	protected Fault getInformation() {
		this.faultyWires = new HashMap<>();
		this.faultyWires.put(this.yRnd, Pair.of(xRnd, xRnd));
		return new Fault(this.getClass().getSimpleName(), this.faultName, this.faultyWires);

	}
}
