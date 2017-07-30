import sys
import os
import tensorflow as tf
import numpy
input_data_path="./data/16-rawdata-max/huajiaoyou-ascii/2016_03_25/merged/"
max_data_file="./data/16-rawdata-max/max.txt"
cat_to_int={'H':[0,0,1],'Q':[0,1,0],'T':[1,0,0]}
#cat_to_int={'H':1,'Q':2,'T':3}
#train_data_file={'H':''}

output_path="./output/"
save_file = output_path+'/bincounter.ckpt'
sess = tf.InteractiveSession()
def max_extract(max_file):
    maxf = open(max_file)
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    group_to_data = dict()
    line = maxf.readline().strip()
    columns = line.split("\t")
    print line
    group_idx = columns.index("Groups") + 1
    line = maxf.readline().strip()
    while line:
        columns = line.split("\t")
        x = columns[1:group_idx]
        y = columns[group_idx][0]
        if not group_to_data.has_key(y):
            group_to_data[y] = [x]
        else:
            group_to_data[y].append(x)
        print "len of ",columns[group_idx],y,len(group_to_data[y])
        line = maxf.readline().strip()
    print "len of group:",len(group_to_data)
    for k,v in group_to_data.items():
        vj = 0
        while vj <len(v)-1:
            train_x.append(v[vj])
            train_y.append(cat_to_int[k])
            vj+=1
        test_x+=v[-1]
        test_y+=cat_to_int[k]
    print "train:",len(train_x),len(train_y)
    print "test:",len(test_x),len(test_y)
    print train_x,train_y
    return [train_x, train_y],[test_x,test_y]

def normalize(data_path):
    files = os.listdir(data_path)
    #print len(files),files
    data_set = dict()
    for f in files:
        simple_cat = f.split('-')[0]
        tmp_f = open(input_data_path + f)
        #print "parse file",tmp_f
        line = tmp_f.readline().strip()
        while line != "[SENSOR DATA]":
            line = tmp_f.readline().strip()
        line = tmp_f.readline().strip()
        one_simple = {
                "cat_name":simple_cat,
                "data":[],
                "file_name":f,
                }
        if not data_set.has_key(simple_cat):
            data_set[simple_cat] = []
        while line:
            columns = line.split()
            #print columns[0],len(columns)
            one_simple["data"].append(columns)
            line = tmp_f.readline().strip()
            if len(columns) != 19:
                tmp_f.close()
                break;
        data_set[simple_cat].append(one_simple)
        #print "data_set_len:%d simple_cat:%s count:%d"%(len(data_set),simple_cat,len(data_set[simple_cat]))
    cat_list = []
    train_x = []
    train_y = []
    max_list = [None]*18
    for simple_cat,values in data_set.items():
        cat = simple_cat[0]
        if not cat_to_int.has_key(cat):
            print "cat not found in set:%s cat:%s"%(cat_to_int,cat)
            sys.exit(1)
        if cat not in cat_list:
            cat_list.append(cat)
            print type(values[0]["data"])
            tmp_data = numpy.matrix(values[0]["data"],dtype=float) #NOTE:max must be numeric type
            max_colume_value = tmp_data.max(0)[:,1:]
            print "max",max_colume_value
            #new_s = tmp_data[:,1:]/max_colume_value[:,numpy.newaxis]
            #for s in new_s:
            for s in values[0]["data"]:
                #print s[1:]
                tmp_s = numpy.matrix(s[1:],dtype=float)
                #print tmp_s
                print "simple",cat,len(tmp_s),len(tmp_s[0,:])
                #new_s = tmp_s/max_colume_value[:,numpy.newaxis]
                new_s = tmp_s/max_colume_value
                #print new_s.tolist()[0]
                #print type(new_s[0])
                #sys.exit()
                train_x.append(new_s.tolist()[0])
                train_y.append(cat_to_int[cat])
    xof = open('train_x','w')
    for tx in train_x:
        xof.write("\t".join(map(lambda x:str(x),tx))+"\n")
    xof.close()
    sys.exit()
    print len(train_x),len(train_x[0])#,train_x
    print len(train_y)#,train_y #train_x  = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], #              [1, 1, 0], [1, 1, 1]]
    #train_y = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
    #              [1, 1, 1], [0, 0, 0]]
    #test_x  = [[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1],
    #               [1, 0, 0], [0, 0, 0], [1, 1, 0]]
    #test_y = [[0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0],
    #               [1, 0, 1], [0, 0, 1], [1, 1, 1]]
    #sys.exit()
    test_x = []
    test_y = []
    return [train_x,train_y],[test_x,test_y]

  
def train(data,neuron_struct):
    #    '''
    #    neuron_struct:[
    #    {"name":"layer0","struct":{18,10,"ts.nn.relu"}},
    #    {"name":"layer1","struct":{10,3,"tf.sigmoid"}},
    #    ]
    #    '''
    w = []
    b = []
    h = []
    train_x = train_data[0]
    train_y = train_data[1]
    input_layer_num = neuron_struct[0]["struct"][0]
    output_layer_num = neuron_struct[-1]["struct"][1]
    x = tf.placeholder(tf.float32, shape=[None, input_layer_num])
    y_ = tf.placeholder(tf.float32, shape=[None,output_layer_num])
    results = tf.placeholder(tf.float32, shape=[None,output_layer_num])
    for one_layer in neuron_struct:
        lname = one_layer["name"]
        lstruct = one_layer["struct"]
        with tf.name_scope(lname):
            pre_layer_num,cur_layer_num=lstruct[0],lstruct[1]
            y_caculator = lstruct[2]
            #print lname,pre_layer_num,cur_layer_num
            tmp_w = tf.truncated_normal([pre_layer_num,cur_layer_num],mean=0.5,stddev=0.8)
            w.append(tf.Variable(tmp_w))
            tmp_b = tf.truncated_normal([cur_layer_num],mean=0.5,stddev=0.8)
            b.append(tf.Variable(tmp_b))
            if len(w) == len(neuron_struct):
                results = tf.matmul(h[-1],w[-1])+b[-1]
            if len(w) == 1:
                tmp_y = tf.matmul(x,w[-1])+b[-1]
                h.append(eval(y_caculator)(tmp_y))
            else:
                tmp_y = tf.matmul(h[-1],w[-1])+b[-1]
                h.append(eval(y_caculator)(tmp_y))
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=results, labels=y_))
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=results, labels=y_))
    
    with tf.name_scope('train'):
        train_step = tf.train.RMSPropOptimizer(0.25, momentum=0.5).minimize(cross_entropy)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0)
        train_ops = optimizer.minimize(cross_entropy)
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables())
    is_converge = False
    for i in range(10001):
      if i%1000 == 0:
        train_error = cross_entropy.eval(feed_dict={x: train_x, y_:train_y})
        print("step %d, training error  %g"%(i, train_error))
        if abs(train_error) < 0.5:
            is_converge = True
            break
      
      sess.run(train_ops, feed_dict={x: train_x, y_: train_y})
    print("Saving neural network to %s.*"%(save_file))
    if is_converge:
        saver = tf.train.Saver()
        saver.save(sess, save_file)
    else:
        print "do not save with no converge"
    return is_converge
def test(results):
    saver = tf.train.Saver()
    saver.restore(sess, save_file)
    print('\nCounting starting with: 0 0 0')
    res = sess.run(results, feed_dict={x: [[0, 0, 0]]})
    print('%g %g %g'%(res[0][0], res[0][1], res[0][2]))
    for i in range(8):
      res = sess.run(results, feed_dict={x: res})
      print('%g %g %g'%(res[0][0], res[0][1], res[0][2]))
if __name__ == "__main__":
    neuron_struct=list([
       {
           "name":"layer1",
           #"struct":[18,8,"tf.nn.sigmod"],
           "struct":[18,15,"tf.nn.relu"],
       },
        {
           "name":"layer3",
           "struct":[15,13,"tf.nn.relu"],
       },
{
           "name":"layer3",
           "struct":[13,9,"tf.nn.relu"],
       },

       {
           "name":"layer3",
           "struct":[9,17,"tf.nn.relu"],
       },

       {
           "name":"layer4",
           "struct":[17,5,"tf.nn.sigmoid"],
       },
       {
           "name":"layer5",
           "struct":[5,3,"tf.nn.softmax"],
       },
   ])
 
    #train_data,test_data = normalize(input_data_path)
    train_data,test_data = max_extract(max_data_file)
    #if os.path.isfile(save_file+".meta"):
    #    test(test_data)
    #else:
    is_converge = False
    run_time = 1
    while not is_converge:
        is_converge = train(train_data,neuron_struct)
        run_time+=1
        print "run_time",run_time
