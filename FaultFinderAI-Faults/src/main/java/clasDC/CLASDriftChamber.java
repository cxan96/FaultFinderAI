package clasDC;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;
import org.datavec.image.data.Image;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import faults.Fault;
import faults.FaultFactory;
import faults.FaultNames;
import lombok.AccessLevel;
import lombok.Getter;
import utils.FaultUtils;

/**
 * 
 * @author m.c.kunkel <br>
 *         a CLASDriftChamber contains 2 superlayers 1-2, 3-4, 5-6. Each
 *         superlayer contains 6 layers and 112 wires. <br>
 *         FaultFactory provides the layers and wire faults.
 * 
 */
@Getter
public class CLASDriftChamber implements CLASFactory {
	private int region;
	private int sector;
	private int nchannels;
	private Image image = null;
	private List<Fault> faultList = null;

	@Getter(AccessLevel.NONE)
	private Map<Integer, Pair<Image, List<Fault>>> dcFaults = null;
	@Getter(AccessLevel.NONE)
	private FaultFactory factory = null;

	public CLASDriftChamber(int region, int sector, int nchannels) {
		if (region > 3 || region < 1 || sector > 6 || sector < 1) {
			throw new IllegalArgumentException("Invalid input: (region, sector), must have values less than"
					+ " (4, 7) and more than (0, 0). Received: (" + region + "," + sector + ")");
		}
		this.region = region;
		this.sector = sector;
		this.nchannels = nchannels;
		init();
		concat();
	}

	private void init() {
		this.dcFaults = new HashMap<>();
		if (this.region == 1) {
			factory = new FaultFactory(1, 10, FaultNames.CHANNEL_ONE, true, true, nchannels);
			dcFaults.put(1, Pair.of(factory.asUnShapedImageMatrix(), factory.getFaultList()));
			factory = new FaultFactory(2, 10, FaultNames.CHANNEL_ONE, true, true, nchannels);
			dcFaults.put(2, Pair.of(factory.asUnShapedImageMatrix(), factory.getFaultList()));

		} else if (this.region == 2) {
			factory = new FaultFactory(3, 10, FaultNames.CHANNEL_ONE, true, true, nchannels);
			dcFaults.put(1, Pair.of(factory.asUnShapedImageMatrix(), factory.getFaultList()));
			factory = new FaultFactory(4, 10, FaultNames.CHANNEL_ONE, true, true, nchannels);
			dcFaults.put(2, Pair.of(factory.asUnShapedImageMatrix(), factory.getFaultList()));

		} else {
			factory = new FaultFactory(5, 10, FaultNames.CHANNEL_ONE, true, true, nchannels);
			dcFaults.put(1, Pair.of(factory.asUnShapedImageMatrix(), factory.getFaultList()));
			factory = new FaultFactory(6, 10, FaultNames.CHANNEL_ONE, true, true, nchannels);
			dcFaults.put(2, Pair.of(factory.asUnShapedImageMatrix(), factory.getFaultList()));

		}

	}

	private void concat() {
		// if (this.region == 1) {
		INDArray a = dcFaults.get(1).getLeft().getImage();
		INDArray b = dcFaults.get(2).getLeft().getImage();
		if (a.rank() != b.rank() || a.size(a.rank() == 3 ? 1 : 2) != b.size(b.rank() == 3 ? 1 : 2)
				|| a.size(a.rank() == 3 ? 2 : 3) != b.size(b.rank() == 3 ? 2 : 3)) {
			throw new IllegalArgumentException("Invalid input: arrays are not of equal rank in addImages()");
		}
		/**
		 * Concat along the rows i.e layers
		 */
		INDArray ret = Nd4j.concat(a.rank() == 3 ? 1 : 2, a, b);
		int rank = ret.rank();
		int rows = ret.size(rank == 3 ? 1 : 2);
		int cols = ret.size(rank == 3 ? 2 : 3);
		int nchannels = ret.size(rank == 3 ? 0 : 1);

		this.image = new Image(ret, nchannels, rows, cols);

		/**
		 * This will modify and append the Fault list
		 */
		List<Fault> aList = dcFaults.get(1).getRight();
		List<Fault> bList = dcFaults.get(2).getRight();

		for (Fault fault : bList) {
			fault.offsetFaultCoodinates(6.0, "y");

			aList.add(fault);
		}
		this.faultList = aList;

	}

	public static void main(String[] args) {
		CLASDriftChamber c = new CLASDriftChamber(1, 1, 3);

		INDArray ret = c.getImage().getImage();
		FaultUtils.draw(c.getImage());

		int rank = ret.rank();
		int rows = ret.size(rank == 3 ? 1 : 2);
		int cols = ret.size(rank == 3 ? 2 : 3);
		int nchannels = ret.size(rank == 3 ? 0 : 1);
		// System.out.println(rank + " " + rows + " " + cols + " " + nchannels);
		System.out.println(ret.shapeInfoToString());

		for (Fault fault : c.getFaultList()) {
			fault.printWireInformation();
		}

	}

}
