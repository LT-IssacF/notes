## 课程来源: [C++11开始的多线程编程](https://www.bilibili.com/video/BV1Ya411q7y4/ "双笙子佯谬") ##
# 0. std::chrono(C++11) #
* 时间点: 2023年9月7日 `chrono::steady_clock::time_point等`
* 时间段: 10秒 `chrono::milliseconds, chrono::seconds, chrono::minutes等`
* 时间的相关计算符合现实标准
```C++
auto t0 = chrono::steady_clock_now(); // 获取当前时间点
auto t1 = t0 + chrono::seconds(30); // 当前时间点的30s后
auto dt = t1 - t0; // 时间差(30s时间段)
int sec = chrono::duration_cast<chrono::seconds>(dt).count(); // 时间段的秒数
```
* duration_cast可以在任意的duration类型间转换，duration<T, R>意思是用T类型表示单位为R的时间，例如：
```C++
using double_ms = std::chrono::duration<double, std::milli>;
double ms = std::chrono::duration_cast<double_ms>(dt).count();
// 将时间差用double的类型表示为毫秒数
```

---
# 1. 进程 #
## std::thread(C++11) ##
使用需在CMakeLists.txt添加：
```cmake
find_package(Threads REQUIRED)
target_link_libraries(main PUBLIC Threads::Threads)
```
* std::this_thread::sleep_for
```C++
std::this_thread::sleep_for(std::chrono::milliseconds(400));
// 让当前线程睡眠400ms，chrono的强类型让单位选择更自由
```
* std::this_thread::sleep_until
```C++
auto t = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
std::this_thread::sleep_until(t);
// 让进程睡到某个时间点
```

---
传统的C++程序只是一个main函数的单线程，必须从头至尾挨个执行，效率低
* 使用lambda构造
```C++
std::thread t1([&] {
    download("hello.zip");
});
```
* 主线程等待子线程结束：`t1.join()`
在return之前添加`t1.join()`可以让主线程等到t1结束再继续进行

---
* std::thread的析构函数会销毁线程
作为一个C++类，thread遵循RAII(Resource Acquision Is Initialization)思想和三五法则(拷贝构造、拷贝赋值、析构、移动构造、移动赋值)  
t1如果定义在其他函数中，如果该函数执行完，那么t1也就跟着被销毁了，这时如果继续使用t1就会出错

* 析构函数不再销毁线程：`t1.detach()`
但是t1并不会自动`join()`，即主线程会不管被分离的t1是否执行完  
解决方案：全局线程池
```C++
class ThreadPool {
public:
    void push_back(std::thread thr) {
        m_pool.push_back(std::move(thr));
    }
    ~ThreadPool() { // main函数结束前自动调用
        for(auto &t : m_pool) // 让线程池中的每一个线程都执行完再return
            t.join();
    }
private:
    std::vector<std::thread> m_pool;
}
```

---
## std::jthread(C++20) ##
jthread的析构函数会自动调用`join()`，如果`joinable()`的话

---
# 2. 异步 #
## std::async ##
    #include <future>
* `std::async`接受一个带返回值的lambda，自身返回一个`std::future`对象，并创建一个线程
```C++
int download(std::string file) {
    for (int i = 0; i < 10; i++) {
        std::cout << "Downloading " << file
                  << " (" << i * 10 << "%)..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    std::cout << "Download complete: " << file << std::endl;
    return 404;
}

void interact() {
    std::string name;
    std::cin >> name;
    std::cout << "Hi, " << name << std::endl;
}

int main() {
    std::future<int> fret = std::async([&] {
        return download("hello.zip"); 
    }); // lambda的函数体将在另一个线程里执行，即download会在后台运行
    interact();
    int ret = fret.get(); // 如果此时download没完成，会等待download完成
    std::cout << "Download result: " << ret << std::endl;
    return 0;
}
```
* `get()`会隐式地等待线程执行完毕
* `wait()`可以显示地等待，只要线程没有执行完，会一直wait下去
* `wait_for()`等待一段时间，返回一个`std::future_status`
```C++
auto stat = fret.wait_for(std::chrono::milliseconds(1000));
if(stat == std::future_status::ready) { // 若在时间内执行完，则返回future_status::ready
    ...
} else { // 如果超过这个时间线程还没有执行完，则放弃等待，返回future_status::timeout
    ...
}
```
* 同理`wait_until()`的参数也是一个时间点

---
## 另一种用法 std::launch::deferred做参数 ##
* `std::async`的第一个参数可设为`std::launch::deferred`，这时不会创建线程，只会把lambda函数体内的运算**推迟**到`future`的`get()`被调用时，所以这种写法不会涉及到多线程，可以实现惰性求值

---
## 手动创建线程：std::promise ##
用`promise`创建线程后，在线程返回时用`set_value()`设置返回值，再在主线程内用`get_future()`获取`future`对象，最后用`get()`获得返回值
```C++
std::promise<int> pret;
std::thread t1([&] {
    auto ret = download("hello.zip");
    pret.set_value(ret);
});
std::future<int> fret = pret.get_future();
interact();
int ret = fret.get();
```

---
# 3. 互斥量 #
## std::mutex ##
```C++
std::vector<int> arr; // vector不是多线程安全容器
std::mutex mtx; // 互斥锁
std::thread t1([&] {
    for(int i = 0; i < 10; i++) {
        mtx.lock();
        arr.push_back(1);
        mtx.unlock();
    }
});
std::thread t2([&] {
    for(int i = 0; i < 10; i++) {
        mtx.lock();
        arr.push_back(2);
        mtx.unlock();
    }
})
```
* `lock()`不仅会给线程上锁，还会检查是否已经上锁，如果已上锁，只有当别的线程`unlock()`后才会继续上锁，否则就会一直等下去
* `try_lock()`成功上锁后会返回`true`，如果已经上锁了则不会无限等待并返回`false`
* `try_lock_for()`可以等待一段时间，如果在时间段内成功上锁则返回`true`，反之亦然，同样的还有`try_lock_until`

---
## std::lock_guard ##
`std::lock_guard`为了防止程序员犯错写成死锁，将对`std::mutex`的`lock()`和`unlock()`操作进行封装，它的构造函数包含前者，析构函数包含后者，故退出作用域时能自动解锁
```C++
std::thread t1([&] {
    for(int i = 0; i < 10; i++) {
        std::lock_guard grd(mtx);
        arr.push_back(1);
    }
});
std::thread t2([&] {
    for(int i = 0; i < 10; i++) {
        std::lock_guard grd(mtx);
        arr.push_back(2);
    }
});
```

---
## std::unique_lock：更高的自由度 ##
`std::lock_guard`严格在析构时`unlock()`，但有时需要提前`unlock()`
```C++
std::thread t1([&] {
    for(int i = 0; i < 10; i++) {
        std::unique_lock grd(mtx);
        arr.push_back(1);
    }
});
std::thread t2([&] {
    for(int i = 0; i < 10; i++) {
        std::unique_lock grd(mtx);
        arr.push_back(2);
        grd.unlock();
        // grd.lock(); // 甚至可以再次上锁
    }
});
```
`std::unique_lock`储存了一个flag表示锁的状态，在析构时会检查，若还锁着，则`unlock()`，即忘记解锁也没关系

## 用std::defer_lock做参数 ##
* `std::unique_lock`还可以用`std::defer_lock`做参数
```C++
std::unique_lock grd(mtx, std::defer_lock);
```
构造函数不会自动`lock()`，而必须手动调用

## 用std::try_to_lock做参数 ##
此时`std::unique_lock`的构造函数不会用`lock()`，而是会用`try_lock`

---
# 4. 死锁 #
```C++
std::thread t1([&] {
    for(int i = 0; i < 10; i++) {
        mtx1.lock(); // 1
        mtx2.lock(); // 3
        mtx2.unlock();
        mtx1.unlock();
    }
});
std::thread t2([&] {
    for(int i = 0; i < 10; i++) {
        mtx2.lock(); // 2
        mtx1.lock(); // 4
        mtx1.unlock();
        mtx2.unlock();
    }
});
```
当程序按如上的顺序执行，两个线程就会互相无限等待下去，即死锁

---
* 最简单的解决办法就是，一个**线程永远不要同时上多个锁**
* 也可以使线程们的上锁顺序一致
* 也可以使用`std::lock()`，它保证了任意数量线程的调用顺序是否相同都不会死锁，如：
```C++
std::thread t1([&] {
    for(int i = 0; i < 10; i++) {
        std::lock(mtx1, mtx2);
        mtx1.unlock();
        mtx2.unlock();
    }
});
std::thread t2([&] {
    for(int i = 0; i < 10; i++) {
        std::lock(mtx2, mtx1);
        mtx2.unlock();
        mtx1.unlock();
    }
});
```
* `std::lock()`的RAII版本`std::scoped_lock()`，退出作用域后自动解锁

---
同一个线程重复调用`lock()`也会死锁
```C++
void other() {
    mtx1.lock();
    mtx1.unlock();
}
void func() {
    mtx1.lock();
    other();
    mtx1.unlock();
}
```
如上，当func调用other时也会造成死锁
* 最好不要在other里面上锁
* 或者可以改用`std::mutex`为`std::recursive_mutex`

---
# 5. 数据结构 #
## mutable ##
```C++
class MTVector {
public:
    void push_back(int val) {
        m_mtx.lock();
        m_arr.push_back(val);
        m_mtx.unlock();
    }
    size_t size() const {
        m_mtx.lock();
        size_t ret = m_arr.size();
        m_mtx.unlock();
        return ret;
    }
private:
    std::vector<int> m_arr;
    mutable std::mutex m_mtx;
}
```
`size()`为const函数，不能改变类成员数据，但是`m_mtx`的上锁解锁会改变，这样会导致编译错误
* 解决方法就是声明`m_mtx`时前面加上`mutable`

---
## 读写锁：std::shared_mutex ##
* 读可以共享，写必须独占，且读与写不能同时进行
`push_back()`会修改数据，对应读；`size()`只读取数据，可以共享
```C++
size_t size() const {
    m_mtx.lock_shared();
    size_t ret = m_arr.size();
    m_mtx.unlock_shared();
    return ret;
}
```
* 正如`std::unique_lock()`针对`lock()`，`lock_shared()`也有RAII的版本——**`std::shared_lock`**，虽然很绕
```C++
void push_back(int val) {
    std::unique_lock grd(m_mtx);
    m_arr.push_back(val);
}
size_t size() const {
    std::shared_lock grd(m_mtx);
    return m_arr.size();
}
```

---
## 访问者模式 ##
暂时没看懂

---
# 6. 条件变量 #
## 等待被唤醒：std::condition_variable ##
```C++
std::condition_variable cv;
std::mutex mtx;

std::thread t1([&] {
    std::unique_lock lck(mtx);
    cv.wait(lck); // 令t1陷入等待
    std::cout << "t1 is awake" << std::endl;
})

std::cout << "notifying..." << std::endl;
cv.notify_one(); // 主线程调用notify_one会唤醒t1
```
* `std::condition_variable`仅支持`std::unique_lock<std::mutex>`作为`wait()`的参数
* 还有`wait_for()`和`wait_until()`函数，分别接受`std::chrono`的时间段和时间点作为参数

```C++
std::condition_variable cv;
std::mutex mtx;
bool ready = false;

std::thread t1([&] {
    std::unique_lock lck(mtx);
    cv.wait(lck, [&] { return ready; }); // t1陷入
    std::cout << "t1 is awake" << std::endl;
});

std::cout << "notifying not ready" << std::endl;
cv.notify_one(); // expr为false，不唤醒

ready = true;
std::cout << "notifying ready" << std::endl; 
cv.notify_one(); // expr为true，唤醒
```
还可以额外指定一个参数变成`cv.wait(lck, expr)`，其中`expr`是个lambda，只有当其返回值为`true`时才唤醒，否则`notify_one()`也没用
* `notify_one()`只能唤醒一个线程，`notify_all()`可以唤醒所有线程
```C++
    std::thread t1([&] {
        std::unique_lock lck(mtx); // unique_lock构造函数会自动上锁，若先t1线程先启动，则线程t2、t3都会卡在这一步
        cv.wait(lck); // 但是wait()会暂时unlock()
        std::cout << "t1 is awake" << std::endl;
    });
    std::thread t2([&] {
        std::unique_lock lck(mtx);
        cv.wait(lck);
        std::cout << "t2 is awake" << std::endl;
    });
    std::thread t3([&] {
        std::unique_lock lck(mtx);
        cv.wait(lck);
        std::cout << "t3 is awake" << std::endl;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    std::cout << "notifying one" << std::endl;
    cv.notify_one();

    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    std::cout << "notifying all" << std::endl;
    cv.notify_all();
```
## 生产者消费者问题 ##
回忆操作系统那门课中的临界资源，只要访问它，都需要上锁和解锁
```C++
template <class T>
class MTQueue {
public:
    T pop() {
        std::unique_lock lck(m_mtx);
        m_cv.wait(lck, [this] { return !m_arr.empty(); });
        T ret = std::move(m_arr.back());
        m_arr.pop_back();
        return ret;
    }
    void push(T val) {
        std::unique_lock lck(m_mtx);
        m_arr.push_back(std::move(val));
        m_cv.notify_one();
    }
    void push_many(std::initializer_list<T> vals) {
        std::unique_lock lck(m_mtx);
        std::copy(std::move_iterator(vals.begin()), std::move_iterator(vals.end()),std::back_insert_iterator(m_arr));
        m_cv.notify_all(); // 当资源被分配完时剩下依旧阻塞的线程仍然会阻塞
    }
private:
    std::condition_variable m_cv;
    std::mutex m_mtx;
    std::vector<T> m_arr;
}
```

---
# 7. 原子操作 #
## std::atomic ##
有时频繁的上锁解锁会使系统花费大量的开销在内核态的转换上，所以需要有一种开销小的办法完成对临界资源的访问修改
```C++
std::atomic<int> counter = 0;
// 无需上锁
std::thread t1([&] {
    for(int i = 0; i < 10; i++)
        counter += 1;
});
std::thread t2([&] {
    for(int i = 0; i < 10; i++)
        counter += 1;
});
```
* 只能使用`+=`或`++`这类运算符才能保证原子性，`counter = counter + 1;` 写法错误，无法保证原子性

`int old = atm.fetch_add(val);`
* `fetch_add()`不仅会增加`atm`的值，还会返回增加前的值

`int old = counter.exchange(3);`
* `exchange(val)`会把`val`写入`counter`，同时返回`counter`的旧值

`bool flag = counter.compare_exchange_strong(old, val);`
* 判断`counter`与`old`是否相等，若不相等则将`counter`写入`old`，否则将`val`写入`counter`
