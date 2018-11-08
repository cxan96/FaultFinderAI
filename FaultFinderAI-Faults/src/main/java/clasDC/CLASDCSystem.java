package clasDC;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.datavec.image.data.Image;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import faults.Fault;
import lombok.AccessLevel;
import lombok.Getter;
import utils.FaultUtils;

/**
 * 
 * @author m.c.kunkel
 *
 *         A CLASDCSystem consists of 3 CLASDCRegions
 * 
 */
@Getter
public class CLASDCSystem implements CLASFactory {

	private int nchannels;
	private Image image = null;
	private List<Fault> faultList = null;

	@Getter(AccessLevel.NONE)
	private Map<Integer, CLASDCRegion> dcRegions = null;

	public CLASDCSystem(int nchannels) {
		if (!(nchannels == 3 || nchannels == 1)) {
			throw new IllegalArgumentException(
					"Invalid input: (nchannels), must have values of" + " 3 or 1. Received: (" + nchannels + ")");
		}
		this.nchannels = nchannels;
		init();
		concat();
	}

	private void init() {
		this.dcRegions = new HashMap<>();
		this.faultList = new ArrayList<>();

		for (int i = 1; i < 4; i++) {
			dcRegions.put(i, new CLASDCRegion(i, this.nchannels));
		}

	}

	private void concat() {
		INDArray a = dcRegions.get(1).getImage().getImage();
		int rank = a.rank();
		int rows = a.size(rank == 3 ? 1 : 2);
		int cols = a.size(rank == 3 ? 2 : 3);
		int nchannels = a.size(rank == 3 ? 0 : 1);
		INDArray ret = a;
		// ret = ret.reshape(ArrayUtil.combine(new int[] { 1 }, ret.shape()));
		for (int i = 2; i < 4; i++) {
			CLASDCRegion d = dcRegions.get(i);
			INDArray b = d.getImage().getImage();
			if (a.rank() != b.rank() || a.size(a.rank() == 3 ? 1 : 2) != b.size(b.rank() == 3 ? 1 : 2)
					|| a.size(a.rank() == 3 ? 2 : 3) != b.size(b.rank() == 3 ? 2 : 3)) {
				throw new IllegalArgumentException("Invalid input: arrays are not of equal rank in addImages()");
			}
			ret = Nd4j.concat(2, ret, b);
			for (Fault fault : d.getFaultList()) {
				fault.offsetFaultCoodinates(72.0 * (i - 1), "y");
				this.faultList.add(fault);
			}

		}
		rank = ret.rank();
		rows = ret.size(rank == 3 ? 1 : 2);
		cols = ret.size(rank == 3 ? 2 : 3);
		nchannels = ret.size(rank == 3 ? 0 : 1);

		this.image = new Image(ret, nchannels, rows, cols);

	}

	public static void main(String[] args) {
		CLASFactory factory = new CLASDCSystem(3);
		INDArray ret = factory.getImage().getImage();

		FaultUtils.draw(factory.getImage());
		int rank = ret.rank();
		int rows = ret.size(rank == 3 ? 1 : 2);
		int cols = ret.size(rank == 3 ? 2 : 3);
		int nchannels = ret.size(rank == 3 ? 0 : 1);
		System.out.println(rank + " " + rows + " " + cols + " " + nchannels);

		for (Fault fault : factory.getFaultList()) {
			fault.printWireInformation();
		}

	}

}
