package clasDC.faults;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;

public class HVChannelFault extends FaultData {

	private List<Pair<Integer, Integer>> hvChannelSegmentation = new ArrayList<Pair<Integer, Integer>>();;

	public HVChannelFault() {
		setuphvChannelSegmentation();
		this.xRnd = ThreadLocalRandom.current().nextInt(0, hvChannelSegmentation.size());
		setFaultName();
		this.faultyWires = getRandomPair();
	}

	public HVChannelFault(FaultNames faultName) {
		setuphvChannelSegmentation();
		this.xRnd = ThreadLocalRandom.current().nextInt(0, hvChannelSegmentation.size());
		setFaultName();
		while (this.faultName != faultName) {
			this.xRnd = ThreadLocalRandom.current().nextInt(0, hvChannelSegmentation.size());
			setFaultName();
		}
		this.faultyWires = getRandomPair();
	}

	public HVChannelFault(int xplace) {
		setuphvChannelSegmentation();
		this.xRnd = xplace;
		setFaultName();
		this.faultyWires = getRandomPair();
	}

	private Map<Integer, Pair<Integer, Integer>> getRandomPair() {
		Map<Integer, Pair<Integer, Integer>> aNewMap = new HashMap<>();
		for (int i = 1; i < 7; i++) {
			aNewMap.put(i, this.hvChannelSegmentation.get(this.xRnd));
		}
		return aNewMap;
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

	@Override
	public Fault getInformation() {
		return new Fault(this.getClass().getSimpleName(), this.faultName, this.faultyWires,
				Pair.of(this.xRnd, this.yRnd));
	}

	private void setFaultName() {
		if (this.xRnd < 4) {
			this.faultName = FaultNames.CHANNEL_ONE;
		} else if (this.xRnd >= 4 && this.xRnd < 7) {
			this.faultName = FaultNames.CHANNEL_TWO;
		} else {
			this.faultName = FaultNames.CHANNEL_THREE;
		}
	}

	public static void main(String[] args) {
		Fault fault = new HVChannelFault(3).getInformation();
		fault.printWireInformation();
	}
}
