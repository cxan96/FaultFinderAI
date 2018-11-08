package testing;

import java.util.concurrent.ThreadLocalRandom;

public class Test {

	public static void main(String[] args) {

		if (!true) {
			int superLayer = ThreadLocalRandom.current().nextInt(1, 7);
			System.out.println(superLayer);

		} else {
			System.out.println("Not today ISIS");
		}
	}

}
