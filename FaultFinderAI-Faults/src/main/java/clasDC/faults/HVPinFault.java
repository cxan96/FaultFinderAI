package clasDC.faults;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.tuple.Pair;

public class HVPinFault extends FaultData {

	private List<Pair<Integer, Integer>> hvPinSegmentation = new ArrayList<Pair<Integer, Integer>>();

	public HVPinFault() {
		setuphvPinSegmentation();
		this.xRnd = ThreadLocalRandom.current().nextInt(0, hvPinSegmentation.size());
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		this.faultyWires = getRandomPair();
		setFaultName();
	}

	public HVPinFault(FaultNames faultName) {
		setuphvPinSegmentation();
		this.xRnd = ThreadLocalRandom.current().nextInt(0, hvPinSegmentation.size());
		this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
		setFaultName();
		while (this.faultName != faultName) {
			this.xRnd = ThreadLocalRandom.current().nextInt(0, hvPinSegmentation.size());
			this.yRnd = ThreadLocalRandom.current().nextInt(1, 7);
			setFaultName();
		}
		this.faultyWires = getRandomPair();

	}

	public HVPinFault(int xplace, int yplace) {
		setuphvPinSegmentation();
		this.xRnd = xplace;
		this.yRnd = yplace;
		setFaultName();
		this.faultyWires = getRandomPair();

	}

	private Map<Integer, Pair<Integer, Integer>> getRandomPair() {
		Map<Integer, Pair<Integer, Integer>> aNewMap = new HashMap<>();
		aNewMap.put(this.yRnd, this.hvPinSegmentation.get(this.xRnd));
		return aNewMap;
	}

	private void setuphvPinSegmentation() {
		hvPinSegmentation.add(Pair.of(1, 8));
		hvPinSegmentation.add(Pair.of(9, 16));
		hvPinSegmentation.add(Pair.of(17, 24));
		hvPinSegmentation.add(Pair.of(25, 32));
		hvPinSegmentation.add(Pair.of(33, 40));
		hvPinSegmentation.add(Pair.of(41, 48));
		hvPinSegmentation.add(Pair.of(49, 56));
		hvPinSegmentation.add(Pair.of(57, 64));
		hvPinSegmentation.add(Pair.of(65, 72));
		hvPinSegmentation.add(Pair.of(73, 80));
		hvPinSegmentation.add(Pair.of(81, 96));
		hvPinSegmentation.add(Pair.of(97, 112));
	}

	@Override
	public Fault getInformation() {
		return new Fault(this.getClass().getSimpleName(), this.faultName, this.faultyWires,
				Pair.of(this.xRnd, this.yRnd));
	}

	private void setFaultName() {
		if (this.xRnd < 10) {
			this.faultName = FaultNames.PIN_SMALL;
		} else {
			this.faultName = FaultNames.PIN_BIG;
		}
	}
}
