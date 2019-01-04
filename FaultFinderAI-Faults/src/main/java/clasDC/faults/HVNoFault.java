package clasDC.faults;

import java.util.HashMap;

import org.apache.commons.lang3.tuple.Pair;

public class HVNoFault extends FaultData {

	public HVNoFault() {
		this.xRnd = 0;
		this.yRnd = 0;
		this.faultName = FaultNames.NOFAULT;
	}

	@Override
	protected Fault getInformation() {
		this.faultyWires = new HashMap<>();
		this.faultyWires.put(this.yRnd, Pair.of(xRnd, xRnd));
		return new Fault(this.getClass().getSimpleName(), this.faultName, this.faultyWires,
				Pair.of(this.xRnd, this.yRnd));
	}
}
