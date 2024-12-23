/*class Main {
    public static void wow(int n) {
        if (n > 1)
            wow(n/3);
        System.out.print(n+" ");
    }
    public static void main(String[] args) {
        wow(40);
    } 
    public int mystery(int a, int b) {
        if (a == 1)
            return b;
        else
            return a * mystery(a - 1, b);
    }
    System.out.print(mystery(4,6));

} */

/* class Main {
    // Recursive method
    public int mystery(int a, int b) {
        if (a == 1)
            return b;
        else
            return a * mystery(a - 1, b);
    }

    // Main method for execution
    public static void main(String[] args) {
        Main obj = new Main();
        System.out.print(obj.mystery(4, 6));
    }
} */

/* class Main {
    public int interesting (int a, int b) {
        if (b == 1)
            return a*a;
        else if (b == 0)
            return a;
        else
            return interesting (a, b - 2) + interesting (a, b - 1);
    }

    public static void main (String[] args) {
        Main obj = new Main();
        System.out.print(obj.interesting(3, 4));
    }
}
*/

// general idea for looping over string
/*
class Main {
	public static void main(String args[]) {
		String str = "Hello";
		for(int i=0;i<str.length();i++) {
			System.out.println("Index "+i+": "+str.substring(i,i+1));
		}
	}
}
*/

class Main {
public static void main(String args[]) {
    String str = "Computer";
    for (int i = 0; i<str.length(); i+=2) {
        System.out.print(str.substring(i,i+1)); 
    }
}
}