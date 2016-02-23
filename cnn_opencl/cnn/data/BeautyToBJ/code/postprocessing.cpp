enum GENDER {MALE,FEMALE};
enum EMOTION {ANGRY,CALM,CONFUSED,DISGUST,HAPPY,SAD,SCARED,SURPRISED,SQUINT,SCREAM};
enum GLASSES {NO_GLASS,GLASS,SUN_GLASS};
enum MASK {NO_MASK,MASK};
struct Result{
    int   gender;
    float age;
    int   emotion;
    int   glasses;
    int   mask;
    float beauty;
};
int softmax(float* output,int len)
{
   int maxInx=-1;
   float maxValue=-1;
   for (int i=0;i<len;i++)
   {
       float tmp=exp(output[i]);
       if(tmp>maxValue)
       {
           maxValue=tmp;
           maxInx=i;
       }
   }
   return maxInx;
}

// input score (0 - 10) -> output score (0 - 100)
float beauty_map_linear(float score){
    assert(score >= 0 && score <= 10);
    float section[11] =     {0, 1,  2,  3,  4,   5,  6,  7,  8,  9,  10};
    float lower_limit[11] = {25, 35, 45, 55, 65, 75, 85, 90, 93, 96, 100};

    int integer = (int)floor(score);
    float fraction = score - (float) integer;

    if (integer == 10) return 100;
    return lower_limit[integer] + fraction * (lower_limit[integer + 1] - lower_limit[integer]);
}

// input score (0 - 10) -> output score (0 - 100)
float beauty_map_linear_standards(float score, bool is_female, bool is_children){
//    assert(score >= 0 && score <= 10);
    float section[11] =     		{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10};

    float lower_limit_male[11] = 	{25, 35, 45, 55, 65, 75, 85, 90, 93, 96, 100};
    float lower_limit_female[11] = 	{30, 40, 50, 60, 70, 80, 88, 92, 94, 96, 100};
    float lower_limit_child[11] = 	{50, 60, 70, 75, 80, 85, 90, 94, 96, 98, 100};

    int integer = (int)floor(score);
    float fraction = score - (float) integer;

    if (integer == 10) return 100;

    float beauty_score;
    if (is_children)
        beauty_score =  lower_limit_child[integer] + fraction * (lower_limit_child[integer + 1] - lower_limit_child[integer]);
    else if (is_female) {
        beauty_score = lower_limit_female[integer] + fraction * (lower_limit_female[integer + 1] - lower_limit_female[integer]);
    } else {
        beauty_score = lower_limit_male[integer] + fraction * (lower_limit_male[integer + 1] - lower_limit_male[integer]);
    }
    return beauty_score;
}

void vec2output(float* output, Result& r)
{
    //gender
    r.gender=softmax(output+0,2);
    //age
    r.age=0;
    for (int i=2;i<102;i++)
        r.age+=output[i];
    r.age -= 1.0;
    // emotion;

    r.emotion=softmax(output+102,10);

    r.glasses=softmax(output+112,3);
    r.mask=softmax(output+115,2);
    // beauty
    r.beauty = 0;
    for (int i =117; i<127;i++){
        r.beauty += output[i];
    }
    r.beauty = beauty_map_linear_standards(r.beauty, r.gender == 1, r.age <= 10);
//    r.beauty = beauty_map_linear(r.beauty);
}

int main(){

    float output_blob[127] = { . . . . .  };

    // Assume output_blob contains the output of the last layer:

    Result r;
    vec2output(buffer_out, r);
    cout << r.gender << endl;
    cout <<  age << endl;
    cout <<    emotion << endl;
    cout <<    glasses << endl;
    cout <<    mask << endl;
    cout <<  beauty << endl;




}
