public class reversal {
    public static void main(String[] args){
        int num = 92348;
        int i = 0;
        while(num > 0) {
            i = i * 10 + num%10;
            num /= 10;
        }
        System.out.println(i);
    }
}