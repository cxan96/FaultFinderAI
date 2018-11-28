package clasDC.faults;

import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;

public abstract class FaultData {
	protected int xRnd;
	protected int yRnd;
	protected Map<Integer, Pair<Integer, Integer>> faultyWires;

	protected abstract Fault getInformation();

	protected FaultNames faultName;

}
